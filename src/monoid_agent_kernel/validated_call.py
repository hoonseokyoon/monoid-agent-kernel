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
import json
from dataclasses import dataclass, replace
from typing import Any, Literal

from monoid_agent_kernel.core.invocation import InvocationContext
from monoid_agent_kernel.core.model_io import ModelCallReceipt
from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputValidator,
    OutputValidatorError,
    build_repair_message,
    run_output_validators,
)
from monoid_agent_kernel.model_call import DeltaConsumer, ModelCallRunner, ShouldAbort
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn

ValidatedCallStatus = Literal["ok", "unsatisfied", "refusal", "truncated"]


@dataclass(frozen=True)
class ValidatedCallResult:
    """What one validated call produced, across every model call it made.

    Exhaustion is a *result* (``status="unsatisfied"``), not an exception -- mirroring the
    loop's ``limited`` settle. Refusal and truncation short-circuit **before** validation, so
    a cut-off-but-parseable answer can never pass as success; both carry the turn so the
    caller can inspect what arrived. ``receipts`` records the original call and every repair
    call, in order: the audit trail is complete even when the outcome is not ``"ok"``.
    """

    status: ValidatedCallStatus
    turn: ModelTurn
    receipts: tuple[ModelCallReceipt, ...]
    value: Any = None
    ok_values: tuple[tuple[str, Any], ...] = ()
    failures_history: tuple[dict[str, Any], ...] = ()
    repair_calls_used: int = 0


@dataclass
class ValidatedCallRunner:
    """Wrap a :class:`ModelCallRunner` with validators and an explicit repair budget.

    ``max_repair_calls`` bounds *repair calls*, not attempts: 1 (the default, matching
    ``RunLimits.max_output_retries``) means one original call plus at most one repair. The
    deadline and cancellation passed to :meth:`acall` span the whole validated call --
    repairs run under the same boundary, not a fresh one each.
    """

    runner: ModelCallRunner
    validators: tuple[OutputValidator, ...] = ()
    max_repair_calls: int = 1

    def __post_init__(self) -> None:
        if self.max_repair_calls < 0:
            raise ValueError("max_repair_calls must not be negative")

    async def acall(
        self,
        request: ModelRequest,
        *,
        context: InvocationContext | None = None,
        deadline: float | None = None,
        should_abort: ShouldAbort | None = None,
        delta_consumer: DeltaConsumer | None = None,
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
                delta_consumer=delta_consumer,
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
            repair_calls += 1
            current = _repair_request(current, turn, build_repair_message(failures))

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
    """

    if request.output_schema is None or not turn.final_text:
        return None
    try:
        return json.loads(turn.final_text)
    except ValueError:
        return None


def _repair_request(request: ModelRequest, turn: ModelTurn, repair_text: str) -> ModelRequest:
    """The follow-up call asking the model to fix its answer.

    ``tools=()`` on every shape: the standalone surface has no tool executor, so a repair turn
    that could request tools would be requesting calls nobody will run -- and allowing them is
    exactly the loop-escalation this mode exists to rule out. ``observations`` are cleared for
    the same reason: they answer tool calls a repair turn cannot make.

    Three shapes, mirroring the adapters' own request dispatch:
    - by-value ``messages``: append the assistant's answer and the repair prompt;
    - a provider continuation handle: repair rides ``instruction`` on ``previous_turn_handle``;
    - one-shot instruction with no handle: synthesize the by-value form.
    """

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
