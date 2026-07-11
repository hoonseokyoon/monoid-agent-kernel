from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar

from monoid_agent_kernel.core.wire_validation import (
    optional_list,
    parse_bool,
    parse_int,
    parse_str,
)

RunStatus = Literal["completed", "failed", "limited"]

_T = TypeVar("_T")


def _coerce_output(value: object, model: type[_T]) -> _T:
    """Return ``value`` typed as ``model`` for ``AgentRunResult.output_as``. Already-an-instance
    passes through; a pydantic model is re-validated via ``model_validate``; a mapping is expanded
    as kwargs (dataclass / simple class). Raises ``TypeError`` if it cannot be coerced. No hard
    dependency on pydantic — the model_validate path is duck-typed."""
    if isinstance(value, model):
        return value
    validate = getattr(model, "model_validate", None)
    if callable(validate):
        return validate(value)
    if isinstance(value, dict):
        return model(**value)
    raise TypeError(
        f"final_output is {type(value).__name__}, which is not a {model.__name__} "
        "and cannot be coerced"
    )


@dataclass(frozen=True)
class AgentArtifact:
    artifact_id: str
    path: str
    kind: str
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentTurnResult:
    """Result of a single ``submit()`` (one user turn settling).

    Non-terminal: the run stays open for further submits. ``proposal_*`` reflect
    the accumulated workspace changes so far (preview), and ``turn_handle`` is the
    continuation handle to thread into the next user turn.
    """

    status: RunStatus
    final_text: str
    proposal_path: Path
    proposal_hash: str
    changed_paths: tuple[str, ...] = ()
    turn_handle: str | None = None
    error: str = ""
    error_code: str = ""
    # The validated/parsed value from a successful output validator (its ``ValidationOutcome.value``),
    # or ``None`` when no validator ran. Process-local — not persisted in the checkpoint.
    final_output: object = None
    # All validators' values keyed by validator id (``final_output`` is the last of these).
    outputs: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)

    def output_as(self, model: type[_T]) -> _T:
        """``final_output`` typed as ``model`` — see :meth:`AgentRunResult.output_as`."""
        return _coerce_output(self.final_output, model)


@dataclass(frozen=True)
class Suspension:
    """Why a non-blocking pump (``AgentLoop.run_until_suspended``) returned control.

    ``settled`` — the model produced final text and the run awaits the next user
    message. ``awaiting_tasks`` — the run parked with no model tool work and
    pending tasks; ``awaiting_task_ids``/``has_external`` describe the hosted
    (hitl/automation) tasks a caller must wait on. ``limited`` — a per-submit or
    session budget was hit. ``terminal`` — cancelled/timed out/failed.
    ``turn_failed`` — the model turn raised a *recoverable* error (e.g. a 4xx/429
    or a gateway-flagged retryable error): the session is **not** terminal, the
    conversation up to the user message is preserved, and a caller may re-issue
    the turn via ``arun_until_suspended(None)`` or park for new user input.
    ``retryable``/``http_status`` carry the classification for that decision.
    ``interrupted`` — an external caller stopped the current turn (a "stop"); like
    ``turn_failed`` the session is **not** terminal (no error), so a caller parks for
    the next user message. ``paused`` — a cooperative pause froze the turn at the start of
    a step; unlike ``interrupted`` the in-flight ``pending_observations`` are kept, so a
    ``run_until_suspended(None)`` re-pump resumes the same turn where it left off. The
    non-terminal-ness is carried by ``reason`` alone — ``status`` mirrors the failure
    (``"failed"``) for ``turn_failed`` since ``RunStatus`` has no non-terminal value, so
    callers must branch on ``reason``, not ``status``, to detect a live run. For every
    reason except ``awaiting_tasks`` a settle checkpoint ran and ``turn`` carries its result.
    """

    reason: Literal[
        "settled", "awaiting_tasks", "limited", "terminal", "turn_failed", "interrupted", "paused"
    ]
    status: RunStatus
    final_text: str = ""
    error: str = ""
    error_code: str = ""
    awaiting_task_ids: tuple[str, ...] = ()
    has_external: bool = False
    turn: AgentTurnResult | None = None
    retryable: bool = False
    http_status: int | None = None


_SUSPENSION_REASONS = frozenset(
    {
        "settled",
        "awaiting_tasks",
        "limited",
        "terminal",
        "turn_failed",
        "interrupted",
        "paused",
    }
)


def suspension_checkpoint_payload(suspension: Suspension) -> dict[str, Any]:
    """Return the portable, JSON-native observation of one durable park boundary.

    ``turn`` is a projection artifact with local paths and metrics. The durable observation keeps
    only the fields a recovery driver needs to return the same boundary result after a restart.
    """

    return {
        "reason": suspension.reason,
        "status": suspension.status,
        "final_text": suspension.final_text,
        "error": suspension.error,
        "error_code": suspension.error_code,
        "awaiting_task_ids": list(suspension.awaiting_task_ids),
        "has_external": suspension.has_external,
        "retryable": suspension.retryable,
        "http_status": suspension.http_status,
    }


def suspension_from_checkpoint_payload(payload: Mapping[str, Any]) -> Suspension:
    """Rebuild a :class:`Suspension` from its durable park observation."""

    reason = parse_str(payload, "reason")
    status = parse_str(payload, "status")
    if reason not in _SUSPENSION_REASONS:
        raise ValueError(f"unsupported durable suspension reason: {reason!r}")
    if status not in {"completed", "failed", "limited"}:
        raise ValueError(f"unsupported durable suspension status: {status!r}")
    raw_task_ids = optional_list(payload, "awaiting_task_ids")
    if any(not isinstance(task_id, str) for task_id in raw_task_ids):
        raise ValueError("durable suspension task ids must be strings")
    raw_http_status = payload.get("http_status")
    http_status = None if raw_http_status is None else parse_int(payload, "http_status")
    return Suspension(
        reason=reason,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        final_text=parse_str(payload, "final_text"),
        error=parse_str(payload, "error"),
        error_code=parse_str(payload, "error_code"),
        awaiting_task_ids=tuple(raw_task_ids),
        has_external=parse_bool(payload, "has_external"),
        retryable=parse_bool(payload, "retryable"),
        http_status=http_status,
    )


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: RunStatus
    final_text: str
    run_dir: Path
    diff_path: Path
    proposal_path: Path
    artifacts: tuple[AgentArtifact, ...] = ()
    final_outputs: tuple[str, ...] = ()
    final_notes: str | None = None
    # The validated/parsed value from a successful output validator (its ``ValidationOutcome.value``),
    # or ``None`` when no validator ran. Process-local — not persisted in the checkpoint.
    final_output: object = None
    # All validators' values keyed by validator id (``final_output`` is the last of these).
    outputs: dict[str, object] = field(default_factory=dict)
    metrics: dict[str, object] = field(default_factory=dict)
    error: str = ""
    error_code: str = ""
    final_turn_handle: str | None = None

    def output_as(self, model: type[_T]) -> _T:
        """``final_output`` typed as ``model`` — restores the static type a validator erased into
        ``object`` (parity with instructor's typed ``response_model`` return). Already-an-instance
        passes through; a pydantic model is re-validated; a mapping is expanded as kwargs."""
        return _coerce_output(self.final_output, model)
