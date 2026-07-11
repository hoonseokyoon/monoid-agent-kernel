from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from monoid_agent_kernel.core.agents import AgentRuntimeConfig
from monoid_agent_kernel.core.control import ControlCommand, ControlResult
from monoid_agent_kernel.core.control_audit import ControlAuditPolicy
from monoid_agent_kernel.core.lifecycle import LoopSession, SessionState
from monoid_agent_kernel.core.lease_admission import sanitize_denied_capability_result
from monoid_agent_kernel.errors import NativeAgentError, PermissionDenied
from monoid_agent_kernel.reference._shared.tokens import TokenError, TokenManager
from monoid_agent_kernel.reference.backend.ports import LoopPort, TokenClaimsPort

_CONTROL_AUDIT_POLICY = ControlAuditPolicy()


class ControlAuditEventPort(Protocol):
    def __call__(
        self,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
        *,
        level: str = "info",
    ) -> None: ...


class CommandRunControlPort(Protocol):
    def pause_run(self, run_id: str, token: str) -> dict[str, Any]: ...

    def signal_resume(self, run_id: str, token: str) -> dict[str, Any]: ...

    def resume_run(self, run_id: str, token: str) -> dict[str, Any]: ...

    def cancel_run(self, run_id: str, token: str) -> dict[str, Any]: ...

    def interrupt_turn(self, run_id: str, token: str) -> dict[str, Any]: ...


class CommandTaskMessagePort(Protocol):
    def report_task_result(
        self,
        run_id: str,
        token: str,
        *,
        task_id: str,
        result: dict[str, Any],
        status: str = "answered",
    ) -> dict[str, Any]: ...

    def send_message(
        self,
        run_id: str,
        token: str,
        content: str | Sequence[Any],
        *,
        message_id: str = "",
        source: str = "api",
        correlation_id: str = "",
        causation_id: str = "",
        traceparent: str = "",
        tracestate: str = "",
        message_type: str = "user_message",
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def create_task(
        self,
        run_id: str,
        token: str,
        *,
        kind: str,
        request: dict[str, Any],
    ) -> dict[str, Any]: ...


class CommandCapabilityPort(Protocol):
    def revoke_capability(
        self,
        run_id: str,
        token: str,
        *,
        capability: str | None = None,
        lease_id: str | None = None,
        before: float | None = None,
        reason: str = "",
    ) -> dict[str, Any]: ...


class CommandProjectionPort(Protocol):
    def status(self, run_id: str, token: str) -> dict[str, Any]: ...


class CommandRuntimeConfigPort(Protocol):
    def runtime_config(self, run_id: str, token: str) -> dict[str, Any]: ...

    def replace_runtime_config(
        self,
        run_id: str,
        token: str,
        *,
        expected_version: int,
        issuer: str,
        reason: str,
        config: AgentRuntimeConfig,
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BackendCommandContext:
    emit_control_audit_event: ControlAuditEventPort
    verify_run_token: Callable[[str, str], TokenClaimsPort]
    verify_task_callback_token: Callable[[str, str, str], None]
    authorize_claim_subject: Callable[[str, TokenClaimsPort], None]
    is_live_run: Callable[[str], bool]
    active_loop_session: Callable[[str, str], tuple[LoopPort, SessionState]]
    run_control: CommandRunControlPort
    task_messages: CommandTaskMessagePort
    capability_control: CommandCapabilityPort
    projection: CommandProjectionPort
    runtime_config: CommandRuntimeConfigPort


class BackendCommandService:
    """Control command dispatch for the RunnerBackend facade."""

    def __init__(self, context: BackendCommandContext) -> None:
        self._context = context

    def dispatch(
        self, command: ControlCommand, *, audit_token_sha256: str = ""
    ) -> ControlResult:
        args = dict(command.args)
        token = str(args.pop("token", "") or "")
        run_id = command.run_id
        ctype = command.type
        command_id = command.command_id or f"control_{uuid.uuid4().hex[:12]}"
        idempotency_key = command.command_id or command_id
        token_sha256 = audit_token_sha256 or (TokenManager.token_sha256(token) if token else "")
        started = time.time()

        audit_authorized = False
        try:
            self.authorize_control_audit_target(run_id, token, command_type=ctype, args=args)
            audit_authorized = True
            self._context.emit_control_audit_event(
                run_id,
                "control.command.received",
                _CONTROL_AUDIT_POLICY.received_payload(
                    command_id=command_id,
                    command_type=ctype,
                    target_run_id=run_id,
                    actor=command.issuer,
                    reason=command.reason,
                    token_sha256=token_sha256,
                    idempotency_key=idempotency_key,
                    args=command.args,
                ),
            )
            result = self.dispatch_control_command(
                command,
                args=args,
                token=token,
                command_id=command_id,
            )
        except PermissionDenied as exc:
            if audit_authorized:
                self._context.emit_control_audit_event(
                    run_id,
                    "control.command.failed",
                    _CONTROL_AUDIT_POLICY.failed_payload(
                        command_id=command_id,
                        command_type=ctype,
                        target_run_id=run_id,
                        actor=command.issuer,
                        idempotency_key=idempotency_key,
                        token_sha256=token_sha256,
                        status="error",
                        error=str(exc),
                        error_code=getattr(exc, "error_code", "permission_denied"),
                        duration_ms=(time.time() - started) * 1000,
                    ),
                    level="warning",
                )
            raise
        except KeyError as exc:
            result = ControlResult(
                run_id=run_id, type=ctype, status="unsupported", error=str(exc), error_code="run_not_found"
            )
        except (ValueError, NativeAgentError) as exc:
            result = ControlResult(
                run_id=run_id,
                type=ctype,
                status="error",
                error=str(exc),
                error_code=getattr(exc, "error_code", "control_error"),
            )

        duration_ms = (time.time() - started) * 1000
        if result.status == "ok":
            self._context.emit_control_audit_event(
                run_id,
                "control.command.completed",
                _CONTROL_AUDIT_POLICY.completed_payload(
                    command_id=command_id,
                    command_type=ctype,
                    target_run_id=run_id,
                    actor=command.issuer,
                    idempotency_key=idempotency_key,
                    token_sha256=token_sha256,
                    status=result.status,
                    result_code=result.error_code or result.status,
                    state=result.state,
                    duration_ms=duration_ms,
                ),
            )
        else:
            self._context.emit_control_audit_event(
                run_id,
                "control.command.failed",
                _CONTROL_AUDIT_POLICY.failed_payload(
                    command_id=command_id,
                    command_type=ctype,
                    target_run_id=run_id,
                    actor=command.issuer,
                    idempotency_key=idempotency_key,
                    token_sha256=token_sha256,
                    status=result.status,
                    error=result.error,
                    error_code=result.error_code,
                    duration_ms=duration_ms,
                ),
                level="warning",
            )
        return result

    def dispatch_control_command(
        self,
        command: ControlCommand,
        *,
        args: dict[str, Any],
        token: str,
        command_id: str,
    ) -> ControlResult:
        run_id = command.run_id
        ctype = command.type

        def ok(data: dict[str, Any], *, state: str | None = None) -> ControlResult:
            return ControlResult(run_id=run_id, type=ctype, status="ok", state=state, data=dict(data))

        if ctype == "pause":
            return ok(self._context.run_control.pause_run(run_id, token))
        if ctype == "resume":
            return ok(
                self._context.run_control.signal_resume(run_id, token)
                if self._context.is_live_run(run_id)
                else self._context.run_control.resume_run(run_id, token)
            )
        if ctype == "cancel":
            return ok(self._context.run_control.cancel_run(run_id, token))
        if ctype in {"approve", "deny"}:
            result = args.get("result") if isinstance(args.get("result"), dict) else {}
            approval_result = dict(result)
            if ctype == "approve":
                approval_result["answer"] = str(args.get("answer") or "Approve")
                approval_result["approved"] = True
            else:
                approval_result = sanitize_denied_capability_result(
                    result,
                    answer=str(args.get("answer") or "Deny"),
                    reason=command.reason or str(args.get("reason") or approval_result.get("reason") or "denied"),
                )
            return ok(
                self._context.task_messages.report_task_result(
                    run_id,
                    token,
                    task_id=str(args.get("task_id") or ""),
                    result=approval_result,
                    status=str(args.get("status") or "answered"),
                )
            )
        if ctype == "interrupt":
            return ok(self._context.run_control.interrupt_turn(run_id, token))
        if ctype in {"inspect", "health"}:
            loop, state = self._context.active_loop_session(run_id, token)
            session = LoopSession(loop, _state=state)
            if ctype == "inspect":
                inspection = session.inspect()
                return ok(inspection.to_json(), state=inspection.state.value)
            health = session.health()
            return ok(health.to_json(), state=health.state.value)
        if ctype == "status":
            return ok(self._context.projection.status(run_id, token))
        if ctype == "runtime_config":
            return ok(self._context.runtime_config.runtime_config(run_id, token))
        if ctype == "replace_runtime_config":
            return ok(
                self._context.runtime_config.replace_runtime_config(
                    run_id,
                    token,
                    expected_version=int(args.get("expected_version", 0)),
                    issuer=command.issuer,
                    reason=command.reason,
                    config=AgentRuntimeConfig.from_json(args["config"]),
                )
            )
        if ctype == "send_message":
            return ok(
                self._context.task_messages.send_message(
                    run_id,
                    token,
                    content=args.get("content") or "",
                    message_id=command_id,
                    source="control",
                )
            )
        if ctype == "create_task":
            return ok(
                self._context.task_messages.create_task(
                    run_id,
                    token,
                    kind=str(args.get("kind") or ""),
                    request=dict(args.get("request") or {}),
                )
            )
        if ctype == "report_task_result":
            return ok(
                self._context.task_messages.report_task_result(
                    run_id,
                    token,
                    task_id=str(args.get("task_id") or ""),
                    result=dict(args.get("result") or {}),
                    status=str(args.get("status") or "answered"),
                )
            )
        if ctype == "revoke_capability":
            before = args.get("before")
            return ok(
                self._context.capability_control.revoke_capability(
                    run_id,
                    token,
                    capability=(str(args["capability"]) if args.get("capability") else None),
                    lease_id=(str(args["lease_id"]) if args.get("lease_id") else None),
                    before=(float(before) if before is not None else None),
                    reason=command.reason,
                )
            )
        return ControlResult(
            run_id=run_id,
            type=ctype,
            status="unsupported",
            error=f"unknown control command type: {ctype}",
            error_code="unknown_control_command",
        )

    def authorize_control_audit_target(
        self,
        run_id: str,
        token: str,
        *,
        command_type: str = "",
        args: dict[str, Any] | None = None,
    ) -> None:
        if _CONTROL_AUDIT_POLICY.accepts_callback_token(command_type):
            try:
                self._context.verify_task_callback_token(
                    run_id, token, str((args or {}).get("task_id") or "")
                )
                return
            except TokenError:
                pass
        claims = self._context.verify_run_token(run_id, token)
        self._context.authorize_claim_subject(run_id, claims)
