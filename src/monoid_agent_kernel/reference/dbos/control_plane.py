"""Finite, partition-serialized DBOS workflows for durable Reference control commands.

This optional profile deliberately does not import or compose the legacy ``LeaseStore``,
``CommandStore``, recovery watchdog, or process-local owner registry. Each command is one
finite DBOS workflow, workflow IDs provide idempotency, and a partitioned queue serializes
commands for the same run while allowing different runs to progress concurrently.

DBOS steps are at-least-once across a crash before their result is checkpointed. A production
dispatcher must therefore retain Monoid's command/effect idempotency and durable outbox rules.
"""

from __future__ import annotations

import math
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, get_args
from urllib.parse import quote

from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.identifiers import namespaced_id
from monoid_agent_kernel.reference._shared.control_transport import (
    COMMAND_RECEIPT_VERSION,
    CommandConflict,
    CommandPrincipal,
    CommandReceipt,
    command_identity_sha256,
    redact_command_credential,
    sanitize_command_data,
)
from monoid_agent_kernel.reference._shared.tokens import TokenManager
from monoid_agent_kernel.reference.dbos.runtime import (
    DbosDependencyError as DbosDependencyError,
    DbosProcessOwnershipError as DbosProcessOwnershipError,
    DbosShutdownTimeout,
    claim_process_owner,
    create_owned_runtime,
    load_dbos,
    release_process_owner,
)

_DBOS_CONTROL_ENVELOPE_VERSION = namespaced_id("dbos-control-envelope.v1")
DBOS_CONTROL_WORKFLOW_NAME = namespaced_id("reference.dbos-control-workflow.v1")
DBOS_CONTROL_STEP_NAME = namespaced_id("reference.dbos-control-step.v1")

DbosControlType = Literal["pause", "resume", "cancel", "status"]
DbosDispatcher = Callable[["DbosControlEnvelope"], ControlResult]


@dataclass(frozen=True)
class DbosControlConfig:
    system_database_url: str
    name: str = "monoid-reference-dbos"
    application_version: str = "monoid-reference-dbos-v1"
    executor_id: str = "local"
    queue_name: str = "monoid-reference-control"
    polling_interval_s: float = 0.05
    shutdown_grace_s: int = 30

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
            isinstance(self.shutdown_grace_s, bool)
            or not isinstance(self.shutdown_grace_s, int)
            or self.shutdown_grace_s < 1
        ):
            raise ValueError("DBOS shutdown_grace_s must be a positive whole number of seconds")


@dataclass(frozen=True)
class DbosControlEnvelope:
    run_id: str
    command_id: str
    type: DbosControlType
    args: dict[str, Any]
    principal: CommandPrincipal
    token_sha256: str = ""
    reason: str = ""
    created_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.run_id or not self.command_id:
            raise ValueError("run_id and command_id are required")
        if self.type not in get_args(DbosControlType):
            raise ValueError(f"unsupported DBOS control command: {self.type!r}")
        sanitized = sanitize_command_data(self.args)
        if sanitized != self.args:
            raise ValueError("DBOS control args must be credential-free before persistence")

    @property
    def identity_sha256(self) -> str:
        return command_identity_sha256(
            command_type=self.type,
            args=self.args,
            principal=self.principal,
            reason=self.reason,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": _DBOS_CONTROL_ENVELOPE_VERSION,
            "run_id": self.run_id,
            "command_id": self.command_id,
            "type": self.type,
            "args": dict(self.args),
            "principal": self.principal.to_json(),
            "token_sha256": self.token_sha256,
            "reason": self.reason,
            "created_at": self.created_at,
            "identity_sha256": self.identity_sha256,
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> DbosControlEnvelope:
        if payload.get("schema_version") != _DBOS_CONTROL_ENVELOPE_VERSION:
            raise ValueError("unsupported DBOS control envelope version")
        principal_payload = payload.get("principal")
        if not isinstance(principal_payload, Mapping):
            raise ValueError("DBOS control principal must be an object")
        args = payload.get("args", {})
        if not isinstance(args, dict):
            raise ValueError("DBOS control args must be an object")
        envelope = cls(
            run_id=str(payload.get("run_id") or ""),
            command_id=str(payload.get("command_id") or ""),
            type=str(payload.get("type") or ""),  # type: ignore[arg-type]
            args=dict(args),
            principal=CommandPrincipal(
                tenant_id=str(principal_payload.get("tenant_id") or ""),
                user_id=str(principal_payload.get("user_id") or ""),
                issuer=str(principal_payload.get("issuer") or ""),
            ),
            token_sha256=str(payload.get("token_sha256") or ""),
            reason=str(payload.get("reason") or ""),
            created_at=float(payload.get("created_at") or 0.0),
        )
        recorded_identity = str(payload.get("identity_sha256") or "")
        if recorded_identity and recorded_identity != envelope.identity_sha256:
            raise ValueError("DBOS control envelope identity mismatch")
        return envelope

    @classmethod
    def from_control_command(
        cls,
        command: ControlCommand,
        *,
        tenant_id: str,
        user_id: str,
    ) -> DbosControlEnvelope:
        if command.type not in get_args(DbosControlType):
            raise ValueError(f"command {command.type!r} is not supported by the DBOS spike")
        if not tenant_id or not user_id:
            raise ValueError("authenticated tenant_id and user_id are required")
        args = dict(command.args)
        token = str(args.pop("token", "") or "")
        command_id = command.command_id or f"control_{uuid.uuid4().hex[:12]}"
        if token and (token in command.run_id or token in command_id):
            raise NativeAgentError(
                "run_id and command_id must not contain the authenticated credential",
                error_code="invalid_command_id",
            )
        # Redact both before and after JSON coercion. Unsupported objects become repr strings
        # during sanitization, and that representation can reintroduce the bearer text.
        execution_args = redact_command_credential(
            sanitize_command_data(redact_command_credential(args, token)),
            token,
        )
        return cls(
            run_id=command.run_id,
            command_id=command_id,
            type=command.type,  # type: ignore[arg-type]
            args=dict(execution_args),
            principal=CommandPrincipal(
                tenant_id=str(redact_command_credential(tenant_id, token)),
                user_id=str(redact_command_credential(user_id, token)),
                issuer=str(redact_command_credential(command.issuer, token)),
            ),
            token_sha256=TokenManager.token_sha256(token),
            reason=str(redact_command_credential(command.reason, token)),
        )


class DbosControlPlane:
    """Host-owned lifecycle wrapper around one DBOS runtime.

    DBOS owns a process-global registry, so this class fails fast when another control plane
    or externally constructed DBOS runtime is active. Tests use subprocess isolation.
    """

    def __init__(self, config: DbosControlConfig, dispatcher: DbosDispatcher) -> None:
        self.config = config
        self._state_lock = threading.Lock()
        self._dispatch_condition = threading.Condition()
        self._active_dispatches = 0
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
            self._workflow = self._register_workflow(dispatcher)
        except Exception:
            runtime = getattr(self, "_runtime", None)
            if runtime is not None:
                runtime.destroy(
                    destroy_registry=True,
                    workflow_completion_timeout_sec=0,
                )
            self._closed = True
            release_process_owner(self._owner_token)
            raise

    @property
    def registered_queue_name(self) -> str:
        """Application-version-scoped DBOS queue name used by this profile."""

        return self._queue_name

    def _register_workflow(self, dispatcher: DbosDispatcher) -> Any:
        runtime = self._runtime

        @runtime.step(name=DBOS_CONTROL_STEP_NAME, retries_allowed=False)
        def dispatch_step(payload: dict[str, Any]) -> dict[str, Any]:
            envelope = DbosControlEnvelope.from_json(payload)
            with self._dispatch_condition:
                self._active_dispatches += 1
            try:
                try:
                    result = dispatcher(envelope)
                except Exception:
                    # DBOS durably stores workflow errors. Keep callback exception text and
                    # chained context outside that persisted/public surface.
                    raise RuntimeError("DBOS command dispatch failed") from None
                if not isinstance(result, ControlResult):
                    raise TypeError("DBOS dispatcher must return ControlResult")
                if result.run_id != envelope.run_id or result.type != envelope.type:
                    raise ValueError("DBOS dispatcher returned a result for a different command")
                return {
                    "result": dict(sanitize_command_data(result.to_json())),
                    "completed_at": time.time(),
                }
            finally:
                with self._dispatch_condition:
                    self._active_dispatches -= 1
                    self._dispatch_condition.notify_all()

        @runtime.workflow(name=DBOS_CONTROL_WORKFLOW_NAME)
        def control_workflow(payload: dict[str, Any]) -> dict[str, Any]:
            envelope = DbosControlEnvelope.from_json(payload)
            dispatch_record = dispatch_step(payload)
            result = dispatch_record["result"]
            status = "completed" if result.get("status") == "ok" else "failed"
            return CommandReceipt(
                run_id=envelope.run_id,
                command_id=envelope.command_id,
                status=status,  # type: ignore[arg-type]
                result=result,
                created_at=envelope.created_at,
                updated_at=float(dispatch_record["completed_at"]),
            ).to_json()

        return control_workflow

    def launch(self) -> None:
        with self._state_lock:
            if self._launched:
                return
            if self._closed:
                raise RuntimeError("a closed DBOS control plane cannot be relaunched")
            try:
                self._preflight_queue_configuration()
                self._runtime.listen_queues([self._queue_name])
                self._runtime.launch()
                # DBOS applies global concurrency per partition. That limit is the
                # cross-executor ordering invariant; worker concurrency is pinned too.
                queue = self._register_control_queue(self._runtime)
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
                queue = self._register_control_queue(client)
            except DBAPIError as exc:
                if _is_uninitialized_queue_table(exc):
                    return
                raise
            self._assert_queue_configuration(queue)
        finally:
            client.destroy()

    def _register_control_queue(self, registrar: Any) -> Any:
        return registrar.register_queue(
            self._queue_name,
            worker_concurrency=1,
            concurrency=1,
            priority_enabled=False,
            partition_queue=True,
            polling_interval_sec=self.config.polling_interval_s,
            on_conflict="always_update",
        )

    def enqueue_control(
        self,
        command: ControlCommand,
        *,
        tenant_id: str,
        user_id: str,
    ) -> CommandReceipt:
        """Persist an authenticated command after removing its bearer credential."""

        if not isinstance(command, ControlCommand):
            raise TypeError("enqueue_control requires a ControlCommand")
        envelope = DbosControlEnvelope.from_control_command(
            command,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        with self._state_lock:
            if not self._accepting:
                raise RuntimeError("DBOS control plane is not accepting commands")
            return self._enqueue_envelope(envelope)

    def _enqueue_envelope(self, command: DbosControlEnvelope) -> CommandReceipt:
        self._require_launched()
        workflow_id = self.workflow_id(command.run_id, command.command_id)
        with self._dbos_module.SetWorkflowID(workflow_id), self._dbos_module.SetEnqueueOptions(
            queue_partition_key=command.run_id
        ):
            handle = self._runtime.enqueue_workflow(
                self._queue_name,
                self._workflow,
                command.to_json(),
            )
        status = handle.get_status()
        self._assert_same_identity(command, status.input)
        return self._receipt_from_status(command.run_id, command.command_id, status)

    def command_receipt(self, run_id: str, command_id: str) -> CommandReceipt:
        self._require_launched()
        handle = self._runtime.retrieve_workflow(self.workflow_id(run_id, command_id))
        status = handle.get_status()
        self._assert_workflow_target(run_id, command_id, status.input)
        return self._receipt_from_status(run_id, command_id, status)

    def wait_for_receipt(
        self,
        run_id: str,
        command_id: str,
        *,
        timeout_s: float = 30.0,
        poll_interval_s: float = 0.01,
    ) -> CommandReceipt:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            receipt = self.command_receipt(run_id, command_id)
            if receipt.status in {"completed", "failed"}:
                return receipt
            time.sleep(poll_interval_s)
        raise TimeoutError(f"DBOS command {command_id!r} did not finish within {timeout_s}s")

    def close(
        self,
        *,
        timeout_s: int | None = None,
    ) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._closing:
                raise RuntimeError("DBOS control plane close is already in progress")
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
        with self._dispatch_condition:
            active_dispatches = self._active_dispatches
        if active_dispatches or active_workflow_ids:
            details = ", ".join(active_workflow_ids[:3])
            if len(active_workflow_ids) > 3:
                details += ", ..."
            suffix = f" Active workflow IDs: {details}." if details else ""
            raise DbosShutdownTimeout(
                "DBOS shutdown grace expired before all command workflows drained; "
                f"terminate the process.{suffix}"
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
        return f"monoid/run/{quote(run_id, safe='')}/control/{quote(command_id, safe='')}"

    @staticmethod
    def versioned_queue_name(queue_name: str, application_version: str) -> str:
        if not queue_name or not application_version:
            raise ValueError("queue_name and application_version are required")
        return (
            f"monoid/control-queue/{quote(queue_name, safe='')}"
            f"/version/{quote(application_version, safe='')}"
        )

    def _require_launched(self) -> None:
        if not self._launched:
            raise RuntimeError("DBOS control plane must be launched before use")

    @staticmethod
    def _assert_queue_configuration(queue: Any) -> None:
        if (
            queue.concurrency != 1
            or queue.worker_concurrency != 1
            or not queue.partition_queue
            or queue.priority_enabled
        ):
            raise RuntimeError("DBOS control queue does not preserve per-run serialization")

    @staticmethod
    def _assert_same_identity(command: DbosControlEnvelope, workflow_input: Any) -> None:
        persisted = DbosControlPlane._envelope_from_workflow_input(workflow_input)
        if persisted.run_id != command.run_id or persisted.command_id != command.command_id:
            raise RuntimeError("DBOS workflow input does not match the submitted command")
        if persisted.identity_sha256 != command.identity_sha256:
            raise CommandConflict(
                f"command_id {command.command_id!r} already belongs to a different command"
            )

    @staticmethod
    def _assert_workflow_target(run_id: str, command_id: str, workflow_input: Any) -> None:
        persisted = DbosControlPlane._envelope_from_workflow_input(workflow_input)
        if persisted.run_id != run_id or persisted.command_id != command_id:
            raise RuntimeError("DBOS workflow input does not match the requested command")

    @staticmethod
    def _envelope_from_workflow_input(workflow_input: Any) -> DbosControlEnvelope:
        if not isinstance(workflow_input, Mapping):
            raise RuntimeError("DBOS workflow did not expose its persisted input")
        args = workflow_input.get("args")
        if not isinstance(args, (list, tuple)) or not args or not isinstance(args[0], Mapping):
            raise RuntimeError("DBOS workflow input has an unexpected shape")
        return DbosControlEnvelope.from_json(args[0])

    @staticmethod
    def _receipt_from_status(run_id: str, command_id: str, status: Any) -> CommandReceipt:
        created_at = float(status.created_at or 0) / 1000.0
        updated_at = float(status.updated_at or status.created_at or 0) / 1000.0
        if status.status == "SUCCESS":
            if not isinstance(status.output, Mapping):
                raise RuntimeError("completed DBOS workflow has an invalid receipt")
            output = status.output
            if output.get("schema_version") != COMMAND_RECEIPT_VERSION:
                raise RuntimeError("completed DBOS workflow has an unsupported receipt version")
            if output.get("run_id") != run_id or output.get("command_id") != command_id:
                raise RuntimeError("completed DBOS workflow receipt target mismatch")
            receipt_status = str(output.get("status") or "")
            if receipt_status not in {"completed", "failed"}:
                raise RuntimeError("completed DBOS workflow has an invalid receipt status")
            result = output.get("result")
            if result is not None and not isinstance(result, Mapping):
                raise RuntimeError("completed DBOS workflow has an invalid result")
            return CommandReceipt(
                run_id=run_id,
                command_id=command_id,
                status=receipt_status,  # type: ignore[arg-type]
                result=dict(result) if isinstance(result, Mapping) else None,
                created_at=float(output.get("created_at") or created_at),
                updated_at=float(output.get("updated_at") or updated_at),
            )
        if status.status in {"ERROR", "MAX_RECOVERY_ATTEMPTS_EXCEEDED", "CANCELLED"}:
            return CommandReceipt(
                run_id=run_id,
                command_id=command_id,
                status="failed",
                result={
                    "status": "error",
                    "error": "DBOS command workflow failed",
                    "error_code": "dbos_workflow_error",
                },
                created_at=created_at,
                updated_at=updated_at,
            )
        return CommandReceipt(
            run_id=run_id,
            command_id=command_id,
            status="pending",
            created_at=created_at,
            updated_at=updated_at,
        )


def _is_uninitialized_queue_table(exc: Exception) -> bool:
    message = str(exc).lower()
    missing_relation = "no such table" in message or "does not exist" in message
    missing_queue_storage = "queues" in message or 'schema "dbos"' in message
    return missing_relation and missing_queue_storage
