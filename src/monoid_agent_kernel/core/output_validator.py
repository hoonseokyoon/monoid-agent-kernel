"""Output validation (final-response conformance).

A developer-supplied :class:`OutputValidator` is checked at the run's settle points; on failure
the engine re-prompts with the validator's feedback, bounded by ``RunLimits.max_output_retries``.
This module defines the integration surface; the orchestration lives in the loop. See
``docs/dx-notes/2026-06-28-output-contract-design.md`` (v1 = post-hoc validate + re-prompt).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from monoid_agent_kernel.core.json_ingress import (
    normalize_json_ingress,
    portable_type_name,
)
from monoid_agent_kernel.core.result import AgentArtifact
from monoid_agent_kernel.errors import NativeAgentError


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

    def __post_init__(self) -> None:
        validate_validation_outcome(self)


def validate_validation_outcome(value: Any) -> ValidationOutcome:
    """Validate a developer-supplied validator result without truthiness coercion."""

    if not isinstance(value, ValidationOutcome):
        raise TypeError(
        f"validate() must return a ValidationOutcome, got {portable_type_name(value)}"
    )
    if type(value.ok) is not bool:
        raise TypeError("ValidationOutcome.ok must be a boolean")
    if type(value.feedback) is not str:
        raise TypeError("ValidationOutcome.feedback must be a string")
    return value


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
    # Best-effort structured view of ``final_text`` when the call carried an
    # ``output_schema`` (W5 ResponseContract) -- a validator must treat it as a convenience,
    # never as the guarantee; the guarantee is the validator itself.
    parsed: Any = None
    # Whether ``parsed`` is a parse *result* at all. ``parsed is None`` cannot answer that:
    # a schema permitting a root ``null`` yields a perfectly valid parsed value of ``None``,
    # indistinguishable from "no schema was requested" and from "the text is not JSON". A
    # validator that rejects on ``parsed is None`` would fail a conforming answer and burn its
    # repair budget on it. This flag is the authority; ``parsed`` is only meaningful when it is
    # ``True``.
    parsed_ok: bool = False


@runtime_checkable
class OutputValidator(Protocol):
    """A developer-supplied check on a run's final response.

    Register validators via ``AgentLoop(output_validators=...)``; a registered validator runs **by
    default** and a per-run ``OutputValidatorBinding(enabled=False)`` in the runtime config disables
    it (default on — the binding is an opt-out, not an opt-in). ``validate`` returns a
    :class:`ValidationOutcome`; it may instead ``raise OutputRetry(feedback)`` (sugar for a
    rejection). A ``ValueError``/``pydantic.ValidationError`` raised from ``validate`` is also
    treated as a rejection (feedback = the message); any *other* exception is a validator defect
    and terminalizes the run as ``output_validator_error`` (no re-prompt).
    """

    @property
    def id(self) -> str:  # noqa: A003 - matches the runtime-config gate key
        ...

    @property
    def schema(self) -> dict | None: ...

    def validate(self, view: FinalOutputView) -> ValidationOutcome: ...


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

    ``receipts`` carries the completed model calls' receipts when the standalone validated
    call raises this (the loop accounts for usage per call as it goes, so its raises leave the
    default). Without it, a defect on attempt N discarded the receipts of every call the
    caller had already paid for.
    """

    error_code = "output_validator_error"
    receipts: tuple[Any, ...] = ()


def _view_with_own_parsed(view: FinalOutputView) -> FinalOutputView:
    """``view`` with a private copy of ``parsed``, copied without recursing.

    ``normalize_json_ingress`` is the kernel's iterative JSON copier -- it exists precisely to
    walk a JSON-domain value "without recursive Python calls" -- and both substitutions are
    turned off here so this is a copy and nothing else: no surrogate repair, no non-finite
    rewrite. ``parsed`` is already strict-ingress output, so both would be no-ops anyway; off
    is what makes that true by construction rather than by coincidence.
    """

    if view.parsed is None:
        return view
    return replace(
        view,
        parsed=normalize_json_ingress(
            view.parsed, substitute_nonfinite=False, normalize_strings=False
        ),
    )


def run_output_validators(
    validators: tuple[OutputValidator, ...], view: FinalOutputView
) -> tuple[list[tuple[str, str]], list[tuple[str, Any]], tuple[str, BaseException] | None]:
    """Run every validator over one view; collect all failures rather than short-circuiting.

    Returns ``(failures, ok_values, defect)``. Exception classification is the protocol
    contract: :class:`OutputRetry`/``ValueError`` (which covers ``pydantic.ValidationError``)
    are rejections carrying feedback; anything else is a validator *defect* reported as
    ``(validator_id, exception)`` -- the model cannot fix a validator bug, so callers must not
    re-prompt on it. Shared by the AgentLoop settle path and the standalone validated call, so
    the classification cannot drift between them.
    """

    failures: list[tuple[str, str]] = []
    ok_values: list[tuple[str, Any]] = []
    for validator in validators:
        # Each validator gets its own copy of the parsed view. The dataclass is frozen but
        # ``parsed`` is one mutable object; shared, one validator's in-place mutation was
        # judged -- and surfaced as a value -- by the next, defeating the documented
        # "read-only" contract.
        #
        # In its own ``try``, and never ``deepcopy``. ``deepcopy`` recurses, and the strict
        # ingress that produced ``parsed`` accepts nesting up to 512 levels -- deep enough to
        # exhaust an ordinary stack -- so a *conforming* answer raised ``RecursionError`` here,
        # outside any classification, before a single validator ran. The kernel's own JSON
        # copier is iterative by construction and cannot, so a legal answer stays validatable
        # instead of becoming a defect. The separate ``try`` matters: inside the one below, a
        # copy failure would hit the ``ValueError``-is-a-rejection rule and be fed back to the
        # model as repair text for a defect it cannot fix.
        try:
            per_view = _view_with_own_parsed(view)
        except Exception as exc:
            return failures, ok_values, (validator.id, exc)
        try:
            outcome = validate_validation_outcome(validator.validate(per_view))
        except OutputRetry as exc:
            outcome = ValidationOutcome(ok=False, feedback=exc.feedback)
        except ValueError as exc:
            outcome = ValidationOutcome(ok=False, feedback=str(exc))
        except Exception as exc:
            return failures, ok_values, (validator.id, exc)
        if outcome.ok:
            ok_values.append((validator.id, outcome.value))
        else:
            failures.append((validator.id, outcome.feedback))
    return failures, ok_values, None


def build_repair_message(failures: Sequence[tuple[str, str]]) -> str:
    """The re-prompt text for a failed validation.

    One dialect of repair for every execution surface: the loop's settle re-prompt and the
    standalone validated call both feed the model exactly this text.
    """

    lines = [
        "Your final response did not satisfy the required output format. "
        "Correct it and respond again:"
    ]
    for validator_id, feedback in failures:
        lines.append(
            f"- ({validator_id}) {feedback}" if feedback else f"- ({validator_id}) invalid output"
        )
    return "\n".join(lines)


def failures_by_validator(history: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Roll a failure history up to per-validator counts (diagnostics surfaces)."""

    counts: dict[str, int] = {}
    for attempt in history:
        for failure in attempt.get("failures", ()):
            vid = str(failure.get("validator_id", ""))
            counts[vid] = counts.get(vid, 0) + 1
    return counts
