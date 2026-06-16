from __future__ import annotations


class NativeAgentError(Exception):
    """Base error for native-agent-runner."""

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
    ) -> None:
        super().__init__(message, error_code=error_code)
        self.provider_error_code = provider_error_code or ""
        self.retryable = retryable
        self.http_status = http_status


class PermissionDenied(NativeAgentError):
    """Raised when a tool call violates capabilities or path policy."""

    error_code = "permission_denied"
    category = "policy"


class ToolPolicyError(NativeAgentError):
    """Raised when a run's tool policy is invalid."""

    error_code = "tool_policy_invalid"
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


def error_code_for_exception(exc: Exception) -> str:
    code = getattr(exc, "error_code", None)
    return str(code) if code else "internal_error"
