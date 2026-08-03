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

ValidatedCallStatus = Literal["ok", "unsatisfied", "refusal", "truncated", "tool_calls"]

@dataclass(frozen=True)
class AttemptStarted:
    """The boundary marker delivered before an attempt's first chunk -- including its zeroth.

    The attempt index alone could not carry the boundary, because it rides *chunks*: an attempt
    that produces none delivered nothing at all, so a consumer holding the previous attempt's
    rejected text was never told to drop it and rendered it beside an ``ok`` result. An attempt
    with no chunks is not exotic -- a stream that ends after its terminal frame carries no text,
    a frameless gateway stream accepted under ``on_unsupported="omit"`` may carry nothing, and a
    non-streaming adapter never emits at all. The marker is delivered for every attempt, whether
    or not a chunk follows, so "the index advanced" is a fact the consumer is *told*, not one it
    has to infer from data that may never come.
    """

    attempt: int


# What an attempt's stream delivers: the boundary marker, then that attempt's chunks.
AttemptEvent = AttemptStarted | ModelStreamChunk

# The streaming consumer for a *validated* call, which is not one stream but one per attempt.
# The attempt index (0 = the original call, 1 = the first repair) rides every event because a
# rejected attempt's text is discarded output: a consumer that renders or accumulates it must
# know, when the index advances, that everything it holds from the previous index is retracted.
# A plain ``DeltaConsumer`` cannot express that -- it would deliver the rejected answer and the
# corrected answer as one undelimited stream -- so this surface takes the wider callable
# instead, and a caller cannot ignore the boundary by accident.
AttemptDeltaConsumer = Callable[[int, AttemptEvent], None]


@dataclass(frozen=True)
class ValidatedCallResult:
    """What one validated call produced, across every model call it made.

    Exhaustion is a *result* (``status="unsatisfied"``), not an exception -- mirroring the
    loop's ``limited`` settle. Refusal, truncation, and a tool-call answer short-circuit
    **before** validation: a cut-off-but-parseable answer can never pass as success, and a
    turn that stopped to request tools has no final answer to judge -- this surface has no
    executor, so ``status="tool_calls"`` hands the turn (calls included) back to the caller
    instead of validating its empty text. All three carry the turn so the caller can inspect
    what arrived. ``receipts`` records the original call and every repair call, in order: the
    audit trail is complete even when the outcome is not ``"ok"``, and an exception escaping
    :meth:`ValidatedCallRunner.acall` carries the completed calls' receipts as its
    ``receipts`` attribute (the failing call's own receipt reaches only
    ``ModelCallRunner.subscriptions`` -- the adapter raised instead of returning it).

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
    :data:`AttemptDeltaConsumer`, opens every attempt with an :class:`AttemptStarted` marker,
    and tags each of that attempt's chunks with the same index -- so a rejected attempt's text
    is never handed to a consumer that cannot tell it was discarded, and the boundary is
    delivered even by an attempt that streams nothing.
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
        try:
            return await self._acall_attempts(
                request,
                receipts,
                history,
                context=context,
                deadline=deadline,
                should_abort=should_abort,
                delta_consumer=delta_consumer,
            )
        except BaseException as exc:
            # Receipts of completed calls must survive the exception: the result carries them
            # on every settled outcome, and an escaping ``OutputValidatorError`` /
            # ``ModelAdapterError`` / boundary error otherwise discarded calls the caller
            # already paid for. Same guarded attribute-stamp pattern as
            # ``mark_provider_retried`` -- an exception that refuses the attribute simply
            # carries none rather than replacing the failure.
            _stamp_receipts(exc, tuple(receipts))
            raise

    async def _acall_attempts(
        self,
        request: ModelRequest,
        receipts: list[ModelCallReceipt],
        history: list[dict[str, Any]],
        *,
        context: InvocationContext | None,
        deadline: float | None,
        should_abort: ShouldAbort | None,
        delta_consumer: AttemptDeltaConsumer | None,
    ) -> ValidatedCallResult:
        repair_calls = 0
        current = request
        while True:
            # Before the call, so the boundary arrives ahead of anything the attempt produces
            # -- and arrives even when it produces nothing.
            if delta_consumer is not None:
                delta_consumer(repair_calls, AttemptStarted(repair_calls))
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
            # A tool-call answer is a distinct outcome, not empty text. This surface has no
            # executor, so there is nothing to do with the calls except hand them back --
            # validating ``final_text or ""`` judged an answer the model never gave, burned a
            # paid repair rewriting it, and recorded an empty assistant message as what the
            # model said (with zero validators it read as a successful ``"ok"``). Both
            # signals are one rule: the calls themselves matter even under a stop_reason a
            # misbehaving adapter or a server-side-tools gateway got wrong.
            if turn.tool_calls or turn.stop_reason == "tool_calls":
                return _result("tool_calls")
            if not self.validators:
                return _result("ok")

            parsed_ok, parsed = _parsed_output(current, turn)
            view = FinalOutputView(
                final_text=turn.final_text or "",
                parsed=parsed,
                parsed_ok=parsed_ok,
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


def _parsed_output(request: ModelRequest, turn: ModelTurn) -> tuple[bool, Any]:
    """Best-effort structured view of the answer, as ``(parsed_ok, parsed)``.

    Two return values rather than one, because the value alone cannot say whether there was a
    parse: an ``output_schema`` permitting a root ``null`` produces a valid parsed value of
    ``None``, which is exactly what "not JSON" and "no schema" also produce. A validator
    following the documented contract and rejecting ``parsed is None`` would reject a
    conforming answer and spend its repair budget on it, so the flag is what a validator reads.

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
        return False, None
    try:
        return True, loads_json_ingress(turn.final_text)
    except ValueError:
        return False, None


def _stamp_receipts(
    error: BaseException, receipts: tuple[ModelCallReceipt, ...]
) -> None:
    """Attach the completed calls' receipts to an escaping exception, best-effort.

    The guarded-setattr pattern of ``mark_provider_retried``: an exception type that refuses
    the attribute (``__slots__``) simply carries no receipts rather than replacing the
    provider's failure with an ``AttributeError``. ``OutputValidatorError`` declares the
    attribute so its carriage is part of the type's contract; everything else gets it stamped.

    An already-stamped exception is appended to, never overwritten: an adapter that
    internally delegates to another validated call propagates an error carrying the *inner*
    call's receipts, and both audit trails are paid calls -- the inner stamp (closest to the
    failure) keeps its place at the front, the outer runner's completed calls follow.
    Overwriting lost the inner trail; skipping lost the outer one.
    """

    existing = getattr(error, "receipts", None) or ()
    try:
        error.receipts = tuple(existing) + tuple(receipts)  # type: ignore[attr-defined]
    except Exception:
        return


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

    Three shapes, chosen by **how the incoming request carried its conversation** -- never by
    what the answer happened to come back with:
    - by-value ``messages``: append the assistant's answer and the repair prompt;
    - a request that itself arrived on a continuation handle: repair rides ``instruction`` on
      the new ``previous_turn_handle``;
    - one-shot instruction with no handle: synthesize the by-value form.

    Keying on the answer's ``response_id`` instead put every one-shot call on the handle path,
    because a provider returns an id for every response it produces. That path is dead on the
    shipped adapters: ``OpenAIModelAdapter`` sends ``store=False`` on every request, so the
    response the repair would continue from was never persisted, and the reference gateway
    inherits it through its opaque handle. Today such a repair never leaves at all -- that
    adapter refuses the by-reference shape outright at its boundary, so a promoted repair fails
    before the call rather than as an opaque provider 404. Either way it is worse than no
    repair: the whole call is lost, receipts included, to an exception.

    ``None`` is the fourth outcome, and it is the honest one: a request that came in **on** a
    continuation handle whose turn came back **without** a new handle has nowhere to continue
    from. The conversation lives on the provider's side of that handle, so synthesizing the
    by-value form would repair against a prompt containing only this turn's instruction --
    losing every prior message, and for an observation-only continuation losing the prompt
    entirely. The caller gets ``unsatisfied`` instead of a confident answer to a different
    question.
    """

    # Each branch carries the conversation exactly one way and clears the other shapes'
    # carriage fields. The shipped adapters would ignore the extras (``messages`` is read
    # first), but the contract never forbids a conforming adapter from *also* reading
    # ``instruction`` on the by-value shape -- a stale one would re-inject the original
    # question -- and every stale field is hashed into the repair's ``request_digest``,
    # making the replay key describe a request no adapter sends.
    if request.messages is not None:
        return replace(
            request,
            tools=(),
            observations=(),
            instruction=None,
            previous_turn_handle=None,
            messages=(
                *request.messages,
                {"role": "assistant", "content": turn.final_text or ""},
                {"role": "user", "content": repair_text},
            ),
        )
    if request.previous_turn_handle:
        if not turn.response_id:
            return None
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
        instruction=None,
        previous_turn_handle=None,
        messages=(
            {"role": "user", "content": request.instruction or ""},
            {"role": "assistant", "content": turn.final_text or ""},
            {"role": "user", "content": repair_text},
        ),
    )
