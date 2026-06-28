"""Output validation (final-response conformance).

A developer-supplied :class:`OutputValidator` is checked at the run's settle points; on failure
the engine re-prompts with the validator's feedback, bounded by ``RunLimits.max_output_retries``.
This module defines the integration surface; the orchestration lives in the loop. See
``docs/dx-notes/2026-06-28-output-contract-design.md`` (v1 = post-hoc validate + re-prompt).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from native_agent_runner.core.result import AgentArtifact
from native_agent_runner.errors import NativeAgentError


@dataclass(frozen=True)
class ValidationOutcome:
    """The result of one :meth:`OutputValidator.validate` call.

    ``ok=True`` accepts the final response (``value`` carries the validated/parsed value, which
    is surfaced as ``AgentRunResult.final_output``); ``ok=False`` rejects it and ``feedback`` is
    the steering text re-prompted to the model. Use :func:`ok`/:func:`retry` for clarity.
    """

    ok: bool
    value: Any = None
    feedback: str = ""


def ok(value: Any = None) -> ValidationOutcome:
    """Accept the final response, optionally carrying a validated/parsed ``value``."""
    return ValidationOutcome(ok=True, value=value)


def retry(feedback: str) -> ValidationOutcome:
    """Reject the final response; ``feedback`` is re-prompted to the model."""
    return ValidationOutcome(ok=False, feedback=feedback)


@dataclass(frozen=True)
class FinalOutputView:
    """Read-only composite view of a run's final output handed to a validator.

    The validator sees the message text *and* all produced files at once (cross-surface checks
    are first-class). ``read_bytes`` reads a workspace file through the workspace's path jail and
    size cap (``RunLimits.max_bytes_read`` by default; pass ``max_bytes`` to raise it for a legit
    large artifact). The final return is always a mandatory ``final_text`` envelope plus optional
    files — never file-only.
    """

    final_text: str
    artifacts: tuple[AgentArtifact, ...] = ()
    final_outputs: tuple[str, ...] = ()
    read_bytes: Callable[..., bytes] = field(default=lambda path, **_: b"")


@runtime_checkable
class OutputValidator(Protocol):
    """A developer-supplied check on a run's final response.

    Register validators via ``AgentLoop(output_validators=...)`` and enable them per run with an
    ``OutputValidatorBinding`` in the runtime config (default off). ``validate`` returns a
    :class:`ValidationOutcome`; it may instead ``raise OutputRetry(feedback)`` (sugar for a
    rejection). A ``ValueError``/``pydantic.ValidationError`` raised from ``validate`` is also
    treated as a rejection (feedback = the message); any *other* exception is a validator defect
    and terminalizes the run as ``output_validator_error`` (no re-prompt).
    """

    @property
    def id(self) -> str:  # noqa: A003 - matches the runtime-config gate key
        ...

    @property
    def schema(self) -> dict | None:
        ...

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        ...


class OutputRetry(Exception):
    """Raise from a validator to reject the final response and re-prompt with ``feedback``.

    Equivalent to returning ``ValidationOutcome(ok=False, feedback=...)``.
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


class OutputValidatorError(NativeAgentError):
    """A validator raised an unexpected exception (a defect, not a rejection).

    The model cannot fix a validator bug, so the run terminalizes rather than re-prompting; the
    exception text is recorded but never fed back to the model.
    """

    error_code = "output_validator_error"
