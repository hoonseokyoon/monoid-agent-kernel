from __future__ import annotations

from typing import Any


class NativeAgentError(Exception):
    """Base error for Monoid Agent Kernel."""

    error_code = "internal_error"
    # Whether the model may usefully retry the failing tool call (e.g. with different
    # arguments). Distinct from ``ModelAdapterError.retryable``, which gates *gateway*
    # transport retries. This flag is informational and surfaced to the model.
    retryable = False
    # Coarse failure family for the model to reason about: "tool" | "policy" |
    # "workspace" | "internal".
    category = "internal"

    def __init__(self, message: str, *, error_code: str | None = None) -> None:
        super().__init__(message)
        if error_code is not None:
            self.error_code = error_code


class ModelAdapterError(NativeAgentError):
    """Raised when the model adapter cannot produce a usable turn."""

    error_code = "model_error"

    def __init__(
        self,
        message: str,
        *,
        error_code: str | None = None,
        provider_error_code: str | None = None,
        retryable: bool = False,
        http_status: int | None = None,
        provider_retried: bool = False,
        config_recoverable: bool = False,
    ) -> None:
        super().__init__(message, error_code=error_code)
        self.provider_error_code = provider_error_code or ""
        self.retryable = retryable
        self.http_status = http_status
        # A refusal the user resolves by changing configuration and resending -- the
        # client-side twin of a provider 4xx, which classifiers treat as "end the turn, keep
        # the session". A client-detected failure (an applied-parameters proof refusal) has no
        # HTTP status to carry that meaning, and without this flag it was classified like an
        # unflagged 5xx: terminal for the whole run. Orthogonal to ``retryable``: resending
        # the same call cannot help (retryable=False), but the session can survive it.
        self.config_recoverable = config_recoverable
        # Whether the adapter's own retry loop ran before giving up. ``retryable`` is a forecast
        # about a *future* attempt; this is a fact about attempts already made, and the two are
        # independent -- an exhausted retry budget leaves a retryable error that will not be
        # retried again. Without it a failed audit record denies retries in exactly the case where
        # they happened most.
        self.provider_retried = provider_retried


class TurnNotSettled(NativeAgentError):
    """A blocking submit facade parked without a settled turn to return.

    Raised by ``AgentLoop.submit`` / ``asubmit`` / ``run_once`` when the turn suspended with
    ``reason="turn_failed"`` (a *recoverable* model-turn failure), ``"interrupted"``, or
    ``"paused"`` — outcomes that produce no ``AgentTurnResult`` because nothing settled. The
    session itself is still alive (``run_once`` closes it in its own ``finally``, as always);
    the non-blocking pump (``run_until_suspended``) hands the same park back as a
    :class:`~monoid_agent_kernel.core.result.Suspension` instead of raising. ``suspension``
    carries the full evidence (reason, error, ``retryable``, ``http_status``,
    ``config_recoverable``) so a driver can decide between re-attempt, config fix, and
    giving up — the same decision the Suspension-reading driver makes.
    """

    error_code = "turn_not_settled"

    def __init__(self, suspension: Any) -> None:
        detail = suspension.error or suspension.reason
        super().__init__(f"turn did not settle ({suspension.reason}): {detail}")
        self.suspension = suspension
        self.reason = suspension.reason
        self.retryable = suspension.retryable
        self.http_status = suspension.http_status


class PermissionDenied(NativeAgentError):
    """Raised when a tool call violates a configured boundary."""

    error_code = "permission_denied"
    category = "policy"


class AgentConfigError(NativeAgentError):
    """Raised when an agent runtime config is invalid for the current run."""

    error_code = "agent_config_invalid"
    category = "policy"


class ToolExecutionError(NativeAgentError):
    """Raised when a tool handler fails in a controlled way."""

    error_code = "tool_handler_error"
    retryable = True
    category = "tool"


class WorkspaceError(NativeAgentError):
    """Raised for invalid or unsafe workspace operations."""

    error_code = "workspace_error"
    category = "workspace"


class RunTimeout(NativeAgentError):
    """Raised when a run exceeds its configured duration limit."""

    error_code = "run_timeout"


class RunCancelled(NativeAgentError):
    """Raised when a run is cancelled by an external caller."""

    error_code = "cancelled"


class TurnInterrupted(NativeAgentError):
    """Raised at a step boundary when the current turn is interrupted by an external
    caller (a "stop"). Unlike :class:`RunCancelled`, this does **not** terminalize the
    run — the loop converts it to a non-terminal ``Suspension(reason="interrupted")`` so
    the session stays alive and the conversation can continue with the next message."""

    error_code = "interrupted"


class TurnPaused(NativeAgentError):
    """Raised at the start-of-step boundary when a cooperative pause is requested. Unlike
    :class:`TurnInterrupted` (a "stop" that abandons the turn and parks for the next user
    message), a pause **freezes the turn in place**: the loop converts it to a non-terminal
    ``Suspension(reason="paused")`` keeping the in-flight ``pending_observations`` intact, so
    a later resume (a ``run_until_suspended(None)`` re-pump) continues the same turn exactly
    where it left off. Pause lands only at the start of the next step, never mid-step."""

    error_code = "paused"


class ModelCallAborted(NativeAgentError):
    """Raised when a caller's ``should_abort`` predicate stops an in-flight model call.

    Distinct from :class:`TurnInterrupted` because the model-call runner is reusable and knows
    nothing about turns: a gateway or a batch driver aborting a stream is not interrupting a
    conversational turn. :class:`~monoid_agent_kernel.loop.AgentLoop` translates this into
    ``TurnInterrupted`` at its own boundary, which is what keeps the session non-terminal — left
    untranslated it would reach the loop's generic failure handler and terminalize the run.
    """

    error_code = "model_call_aborted"


def error_code_for_exception(exc: Exception) -> str:
    code = getattr(exc, "error_code", None)
    return str(code) if code else "internal_error"
