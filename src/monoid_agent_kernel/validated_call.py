"""One validated model call: dispatch, check the answer, repair at most N times.

This is the standalone execution mode for :class:`OutputValidator` (W5, scope doc
``docs/dx-notes/2026-08-02-v0.21-contract-replay-scope.md`` Track A): a caller invoking
:class:`ModelCallRunner` directly -- an LLM-only skill, a gateway, a batch driver -- gets the
same validate-and-re-prompt guarantee the AgentLoop applies at its settle points, without an
agent loop around it.

The boundary this surface exists to hold (v0.20 scope doc §4-3): **a validation failure never
escalates into a tool loop.** Inside ``AgentLoop`` a repair turn is a full agent turn with the
whole bound tool catalog; here a repair call strips ``tools`` entirely and there is no executor
behind it, so the model can only rewrite its answer. Validation logic, exception
classification, and the repair text are imported from ``core.output_validator`` -- the loop
consumes the same functions, so the two surfaces cannot drift.

It lives at the package root beside ``model_call`` for the same reason ``model_call`` does:
it names the provider vocabulary it drives (``ModelRequest``/``ModelTurn``), and ``core``
does not import ``providers``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.json_ingress import loads_json_ingress
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputValidator,
    OutputValidatorError,
    build_repair_message,
    run_output_validators,
)
from monoid_agent_kernel.model_call import DeltaConsumer, ModelCallRunner, ShouldAbort
from monoid_agent_kernel.providers.base import ModelRequest, ModelStreamChunk, ModelTurn

ValidatedCallStatus = Literal["ok", "unsatisfied", "refusal", "truncated"]

# The streaming consumer for a *validated* call, which is not one stream but one per attempt.
# The attempt index (0 = the original call, 1 = the first repair) rides every chunk because a
# rejected attempt's text is discarded output: a consumer that renders or accumulates it must
# know, when the index advances, that everything it holds from the previous index is retracted.
# A plain ``DeltaConsumer`` cannot express that -- it would deliver the rejected answer and the
# corrected answer as one undelimited stream -- so this surface takes the wider callable
# instead, and a caller cannot ignore the boundary by accident.
AttemptDeltaConsumer = Callable[[int, ModelStreamChunk], None]


@dataclass(frozen=True)
class ValidatedCallResult:
    """What one validated call produced, across every model call it made.

    Exhaustion is a *result* (``status="unsatisfied"``), not an exception -- mirroring the
    loop's ``limited`` settle. Refusal and truncation short-circuit **before** validation, so
    a cut-off-but-parseable answer can never pass as success; both carry the turn so the
    caller can inspect what arrived. ``receipts`` records the original call and every repair
    call, in order: the audit trail is complete even when the outcome is not ``"ok"``.

    ``repair_calls_used < max_repair_calls`` on an ``"unsatisfied"`` result means the budget was
    not the limit: the answer was unrepairable without repairing against a different
    conversation (see :func:`_repair_request`), so no repair was attempted.
    """

    status: ValidatedCallStatus
    turn: ModelTurn
    receipts: tuple[ModelCallReceipt, ...]
    value: Any = None
    ok_values: tuple[tuple[str, Any], ...] = ()
    failures_history: tuple[dict[str, Any], ...] = ()
    repair_calls_used: int = 0


@dataclass(frozen=True)
class ValidatedCallRunner:
    """Wrap a :class:`ModelCallRunner` with validators and an explicit repair budget.

    Frozen, because a budget checked once at construction is not a budget if the object can be
    reconfigured afterwards: a reusable runner whose ``max_repair_calls`` is reassigned to
    ``nan`` walks straight past ``__post_init__`` into an unbounded sequence of paid repair
    calls. Immutability makes the one check a guarantee instead of a hope, and it covers the
    other two fields for free. Reconfigure with :func:`dataclasses.replace`, which revalidates.

    ``max_repair_calls`` bounds *repair calls*, not attempts: 1 (the default, matching
    ``RunLimits.max_output_retries``) means one original call plus at most one repair. The
    deadline and cancellation passed to :meth:`acall` span the whole validated call --
    repairs run under the same boundary, not a fresh one each.

    Streaming is per attempt, not per call: :meth:`acall` takes an
    :data:`AttemptDeltaConsumer` and tags every chunk with the attempt that produced it, so a
    rejected attempt's text is never handed to a consumer that cannot tell it was discarded.
    """

    runner: ModelCallRunner
    validators: tuple[OutputValidator, ...] = ()
    max_repair_calls: int = 1

    def __post_init__(self) -> None:
        # Exact int, like every other budget control in the kernel (``RunLimits``,
        # ``ModelRetryConfig.max_attempts``). ``< 0`` alone let three shapes through from
        # dynamically typed configuration, and the loop bound is ``repair_calls >= budget``:
        # ``nan`` makes that comparison permanently false and ``inf`` never reaches it, so an
        # unbounded sequence of *paid* model calls follows from one bad config value. A float
        # like 1.5 quietly rounds the stated bound up, and ``True`` is 1 by accident rather
        # than by intent. A budget is a count.
        if type(self.max_repair_calls) is not int or self.max_repair_calls < 0:
            raise ValueError("max_repair_calls must be a non-negative integer")

    async def acall(
        self,
        request: ModelRequest,
        *,
        context: InvocationContext | None = None,
        deadline: float | None = None,
        should_abort: ShouldAbort | None = None,
        delta_consumer: AttemptDeltaConsumer | None = None,
    ) -> ValidatedCallResult:
        receipts: list[ModelCallReceipt] = []
        history: list[dict[str, Any]] = []
        repair_calls = 0
        current = request
        while True:
            turn, receipt = await self.runner.acall(
                current,
                context=context,
                deadline=deadline,
                should_abort=should_abort,
                delta_consumer=_attempt_scoped(delta_consumer, repair_calls),
            )
            receipts.append(receipt)

            def _result(
                status: ValidatedCallStatus,
                *,
                value: Any = None,
                ok_values: tuple[tuple[str, Any], ...] = (),
            ) -> ValidatedCallResult:
                return ValidatedCallResult(
                    status=status,
                    turn=turn,
                    receipts=tuple(receipts),
                    value=value,
                    ok_values=ok_values,
                    failures_history=tuple(history),
                    repair_calls_used=repair_calls,
                )

            # Refusal and truncation precede validation, in this order, exactly as the loop's
            # settle decision orders them: validating a truncated answer is how a cut-off but
            # well-formed prefix passes as success.
            if turn.stop_reason == "refusal":
                return _result("refusal")
            if turn.stop_reason == "length":
                return _result("truncated")
            if not self.validators:
                return _result("ok")

            view = FinalOutputView(
                final_text=turn.final_text or "",
                parsed=_parsed_output(current, turn),
            )
            failures, ok_values, defect = await asyncio.to_thread(
                run_output_validators, self.validators, view
            )
            if defect is not None:
                validator_id, exc = defect
                raise OutputValidatorError(
                    f"output validator {validator_id!r} raised: {exc}"
                ) from exc
            if not failures:
                value = ok_values[-1][1] if ok_values else None
                return _result("ok", value=value, ok_values=tuple(ok_values))
            history.append(
                {
                    "attempt": len(history) + 1,
                    "failures": [
                        {"validator_id": validator_id, "feedback": feedback}
                        for validator_id, feedback in failures
                    ],
                }
            )
            if repair_calls >= self.max_repair_calls:
                return _result("unsatisfied")
            repair = _repair_request(current, turn, build_repair_message(failures))
            if repair is None:
                # The answer is unrepairable without changing which conversation is being
                # repaired (see ``_repair_request``). Stopping here rather than repairing
                # against a different prompt: an ``unsatisfied`` answer is a result the caller
                # can act on, a confidently-repaired answer to a prompt the model never saw is
                # not. ``repair_calls_used < max_repair_calls`` on an ``unsatisfied`` result is
                # the observable signal that this happened.
                return _result("unsatisfied")
            repair_calls += 1
            current = repair

    def call(
        self,
        request: ModelRequest,
        *,
        context: InvocationContext | None = None,
        deadline: float | None = None,
        should_abort: ShouldAbort | None = None,
    ) -> ValidatedCallResult:
        """Sync facade over :meth:`acall` for callers with no event loop of their own.

        Streaming consumers are inherently async, so ``delta_consumer`` is only on
        :meth:`acall`. Refuses to run inside an active event loop rather than deadlocking it.
        """

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.acall(
                    request,
                    context=context,
                    deadline=deadline,
                    should_abort=should_abort,
                )
            )
        raise RuntimeError(
            "ValidatedCallRunner.call cannot run inside an active event loop; await acall instead"
        )


def _parsed_output(request: ModelRequest, turn: ModelTurn) -> Any:
    """Best-effort structured view of the answer for :attr:`FinalOutputView.parsed`.

    Populated whenever the call carried an ``output_schema`` and the text parses as JSON --
    deliberately NOT gated on the adapter declaring native support, because a best-effort
    provider often returns parseable JSON too and the validator (not this convenience) is the
    guarantee either way. ``None`` for prose; the validator sees the raw text regardless.

    Parsed through the kernel's strict ingress rather than bare ``json.loads``, which accepts
    Python's non-standard ``NaN`` / ``Infinity`` constants: a validator reading ``parsed`` --
    a schema validator will happily call ``NaN`` a number -- would then accept an answer that
    is not JSON at all. One rule for what counts as JSON here, and it is the rule the rest of
    the kernel already applies (which also rules out duplicate keys, unbounded integers, and
    runaway nesting). Anything it refuses leaves ``parsed`` at ``None``, which is the same
    answer prose gets.
    """

    if request.output_schema is None or not turn.final_text:
        return None
    try:
        return loads_json_ingress(turn.final_text)
    except ValueError:
        return None


def _attempt_scoped(
    consumer: AttemptDeltaConsumer | None, attempt: int
) -> DeltaConsumer | None:
    """Bind one attempt's index onto the caller's consumer.

    ``ModelCallRunner`` streams into a plain ``DeltaConsumer`` that knows nothing about
    attempts, so the index is bound here, once per call, rather than left for the consumer to
    infer from chunk order it cannot see the boundaries of.
    """

    if consumer is None:
        return None
    return lambda chunk: consumer(attempt, chunk)


def _repair_request(
    request: ModelRequest, turn: ModelTurn, repair_text: str
) -> ModelRequest | None:
    """The follow-up call asking the model to fix its answer, or ``None`` when there isn't one.

    ``tools=()`` on every shape: the standalone surface has no tool executor, so a repair turn
    that could request tools would be requesting calls nobody will run -- and allowing them is
    exactly the loop-escalation this mode exists to rule out. ``observations`` are cleared for
    the same reason: they answer tool calls a repair turn cannot make.

    Three shapes, mirroring the adapters' own request dispatch:
    - by-value ``messages``: append the assistant's answer and the repair prompt;
    - a provider continuation handle: repair rides ``instruction`` on ``previous_turn_handle``;
    - one-shot instruction with no handle: synthesize the by-value form.

    ``None`` is the fourth outcome, and it is the honest one: a request that came in **on** a
    continuation handle whose turn came back **without** a new handle has nowhere to continue
    from. The conversation lives on the provider's side of that handle, so synthesizing the
    by-value form would repair against a prompt containing only this turn's instruction --
    losing every prior message, and for an observation-only continuation losing the prompt
    entirely. The caller gets ``unsatisfied`` instead of a confident answer to a different
    question.
    """

    if request.messages is None and not turn.response_id and request.previous_turn_handle:
        return None
    if request.messages is not None:
        return replace(
            request,
            tools=(),
            observations=(),
            messages=(
                *request.messages,
                {"role": "assistant", "content": turn.final_text or ""},
                {"role": "user", "content": repair_text},
            ),
        )
    if turn.response_id:
        return replace(
            request,
            tools=(),
            observations=(),
            previous_turn_handle=turn.response_id,
            instruction=repair_text,
        )
    return replace(
        request,
        tools=(),
        observations=(),
        messages=(
            {"role": "user", "content": request.instruction or ""},
            {"role": "assistant", "content": turn.final_text or ""},
            {"role": "user", "content": repair_text},
        ),
    )
