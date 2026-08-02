"""W5 PR 3: the standalone validated model call.

The boundary pins here are the contract (v0.20 scope doc §4-3): a repair call NEVER carries
tools, and refusal/truncation short-circuit BEFORE any validator runs. The repair text must be
byte-identical to the loop's -- both import ``build_repair_message`` from
``core.output_validator``, and the parity test is the observable proof of that sharing.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field

import pytest

from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputValidatorError,
    ValidationOutcome,
    build_repair_message,
)
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, TextDelta
from monoid_agent_kernel.providers.fake import FakeModelAdapter, FakeStreamingModelAdapter
from monoid_agent_kernel.validated_call import ValidatedCallResult, ValidatedCallRunner


@dataclass
class _Validator:
    id: str = "wants-json"
    schema: dict | None = None
    require: str = "{"
    calls: int = 0
    boom: BaseException | None = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        self.calls += 1
        if self.boom is not None:
            raise self.boom
        if view.final_text.startswith(self.require):
            return ValidationOutcome(ok=True, value={"text": view.final_text})
        return ValidationOutcome(ok=False, feedback="answer must be a JSON object")


@dataclass
class _Harness:
    adapter: FakeModelAdapter
    runner: ValidatedCallRunner
    validator: _Validator
    request: ModelRequest = field(
        default_factory=lambda: ModelRequest(instruction="answer", system_prompt="sys", tools=())
    )

    def run(self) -> ValidatedCallResult:
        return asyncio.run(self.runner.acall(self.request))


def _harness(
    turns: list[ModelTurn],
    *,
    validator: _Validator | None = None,
    max_repair_calls: int = 1,
    request: ModelRequest | None = None,
) -> _Harness:
    adapter = FakeModelAdapter(turns=turns)
    validator = validator if validator is not None else _Validator()
    harness = _Harness(
        adapter=adapter,
        runner=ValidatedCallRunner(
            runner=ModelCallRunner(adapter=adapter),
            validators=(validator,),
            max_repair_calls=max_repair_calls,
        ),
        validator=validator,
    )
    if request is not None:
        harness.request = request
    return harness


def test_valid_answer_returns_ok_with_one_receipt() -> None:
    h = _harness([ModelTurn(final_text='{"a": 1}', stop_reason="stop")])
    result = h.run()
    assert result.status == "ok"
    assert result.value == {"text": '{"a": 1}'}
    assert result.ok_values == (("wants-json", {"text": '{"a": 1}'}),)
    assert len(result.receipts) == 1
    assert result.repair_calls_used == 0
    assert result.failures_history == ()


def test_repair_call_strips_tools_and_uses_the_loops_repair_text() -> None:
    """THE boundary pin: tools=() on the repair request, and the repair prompt is exactly
    what the loop would send (shared build_repair_message)."""

    from monoid_agent_kernel.tools.base import ToolSpec

    tool = ToolSpec(
        id="fs.read",
        description="read",
        input_schema={"type": "object"},
        capability="fs.read",
        side_effect="read",
        handler=lambda *_a: None,
    )
    request = ModelRequest(instruction="answer", system_prompt="sys", tools=(tool,))
    h = _harness(
        [
            ModelTurn(final_text="prose, not json", stop_reason="stop", response_id=None),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    result = h.run()

    assert result.status == "ok"
    assert result.repair_calls_used == 1
    assert len(h.adapter.requests) == 2
    repair = h.adapter.requests[1]
    assert repair.tools == ()
    assert repair.observations == ()
    expected_text = build_repair_message([("wants-json", "answer must be a JSON object")])
    assert repair.messages is not None
    assert repair.messages[-1] == {"role": "user", "content": expected_text}
    assert repair.messages[-2] == {"role": "assistant", "content": "prose, not json"}


def test_repair_uses_the_provider_handle_when_the_request_itself_was_handle_based() -> None:
    request = ModelRequest(
        instruction="answer",
        system_prompt="sys",
        tools=(),
        previous_turn_handle="resp_0",
    )
    h = _harness(
        [
            ModelTurn(final_text="nope", stop_reason="stop", response_id="resp_1"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    result = h.run()
    assert result.status == "ok"
    repair = h.adapter.requests[1]
    assert repair.previous_turn_handle == "resp_1"
    assert repair.instruction == build_repair_message(
        [("wants-json", "answer must be a JSON object")]
    )
    assert repair.tools == ()
    assert repair.messages is None


def test_a_one_shot_repair_never_rides_a_handle_the_request_did_not_have() -> None:
    """The shape is chosen by how the request carried its conversation, not by what the answer
    came back with. Providers return a response id for every response, so keying on it put
    every one-shot call on the handle path -- which is dead on the shipped adapters, because
    ``OpenAIModelAdapter`` sends ``store=False`` and the id was never persisted."""

    h = _harness(
        [
            ModelTurn(final_text="prose", stop_reason="stop", response_id="resp_1"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ]
    )
    result = h.run()

    assert result.status == "ok"
    repair = h.adapter.requests[1]
    assert repair.previous_turn_handle is None
    assert repair.messages is not None and len(repair.messages) == 3
    assert repair.messages[0] == {"role": "user", "content": "answer"}
    assert repair.messages[1] == {"role": "assistant", "content": "prose"}
    assert repair.tools == ()


def test_repair_appends_to_by_value_messages() -> None:
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(),
        messages=({"role": "user", "content": "answer"},),
    )
    h = _harness(
        [
            ModelTurn(final_text="nope", stop_reason="stop", response_id="resp_ignored"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    assert h.run().status == "ok"
    repair = h.adapter.requests[1]
    assert repair.messages is not None and len(repair.messages) == 3
    assert repair.messages[0] == {"role": "user", "content": "answer"}


def test_budget_exhaustion_is_a_result_not_an_exception() -> None:
    h = _harness(
        [
            ModelTurn(final_text="wrong 1", stop_reason="stop"),
            ModelTurn(final_text="wrong 2", stop_reason="stop"),
        ],
        max_repair_calls=1,
    )
    result = h.run()
    assert result.status == "unsatisfied"
    assert result.repair_calls_used == 1
    assert len(result.receipts) == 2
    assert [entry["attempt"] for entry in result.failures_history] == [1, 2]
    assert result.turn.final_text == "wrong 2"


def test_zero_repair_budget_never_dispatches_a_second_call() -> None:
    h = _harness([ModelTurn(final_text="wrong", stop_reason="stop")], max_repair_calls=0)
    result = h.run()
    assert result.status == "unsatisfied"
    assert len(h.adapter.requests) == 1


@pytest.mark.parametrize(
    ("stop_reason", "status"),
    [("refusal", "refusal"), ("length", "truncated")],
)
def test_stop_reason_short_circuits_before_any_validator_runs(
    stop_reason: str, status: str
) -> None:
    h = _harness([ModelTurn(final_text="partial", stop_reason=stop_reason)])  # type: ignore[arg-type]
    result = h.run()
    assert result.status == status
    assert h.validator.calls == 0
    assert result.repair_calls_used == 0


def test_validator_defect_raises_and_never_reprompts() -> None:
    h = _harness(
        [
            ModelTurn(final_text="anything", stop_reason="stop"),
            ModelTurn(final_text="never dispatched", stop_reason="stop"),
        ],
        validator=_Validator(boom=RuntimeError("bug in validator")),
    )
    with pytest.raises(OutputValidatorError, match="'wants-json' raised"):
        h.run()
    assert len(h.adapter.requests) == 1


def test_value_error_is_a_rejection_with_feedback_not_a_defect() -> None:
    h = _harness(
        [
            ModelTurn(final_text="first", stop_reason="stop"),
            ModelTurn(final_text="second", stop_reason="stop"),
        ],
        validator=_Validator(boom=ValueError("not parseable")),
    )
    result = h.run()
    assert result.status == "unsatisfied"
    assert result.failures_history[0]["failures"][0]["feedback"] == "not parseable"


def test_sync_facade_works_and_refuses_a_running_loop() -> None:
    h = _harness([ModelTurn(final_text='{"a": 1}', stop_reason="stop")])
    result = h.runner.call(h.request)
    assert result.status == "ok"

    async def _inside() -> None:
        with pytest.raises(RuntimeError, match="active event loop"):
            h.runner.call(h.request)

    asyncio.run(_inside())


def test_no_validators_means_plain_dispatch() -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(final_text="hello", stop_reason="stop")])
    runner = ValidatedCallRunner(runner=ModelCallRunner(adapter=adapter))
    result = asyncio.run(
        runner.acall(ModelRequest(instruction="hi", system_prompt="s", tools=()))
    )
    assert result.status == "ok"
    assert result.value is None
    assert len(result.receipts) == 1


@pytest.mark.parametrize(
    "budget",
    [
        -1,
        float("nan"),  # ``repair_calls >= nan`` is never true: unbounded paid repair calls
        float("inf"),  # never reached either
        1.5,  # a fractional budget rounds the stated bound up
        True,  # an int by accident, not by intent
        "1",
        None,
    ],
)
def test_only_an_exact_non_negative_int_budget_is_accepted(budget: object) -> None:
    """A budget is a count. Everything here reached the loop bound from dynamically typed
    configuration, and two of them made it permanently unsatisfiable."""

    with pytest.raises(ValueError, match="max_repair_calls"):
        ValidatedCallRunner(
            runner=ModelCallRunner(adapter=FakeModelAdapter()),
            max_repair_calls=budget,  # type: ignore[arg-type]
        )


def test_zero_and_positive_int_budgets_are_still_accepted() -> None:
    for budget in (0, 1, 7):
        runner = ValidatedCallRunner(
            runner=ModelCallRunner(adapter=FakeModelAdapter()), max_repair_calls=budget
        )
        assert runner.max_repair_calls == budget


def test_the_budget_cannot_be_reassigned_past_the_check() -> None:
    """A budget checked once at construction is not a budget if the object can be
    reconfigured afterwards -- a reusable runner reassigned to nan walks straight past
    __post_init__ into unbounded paid repair calls. The runner is frozen."""

    runner = ValidatedCallRunner(runner=ModelCallRunner(adapter=FakeModelAdapter()))
    with pytest.raises(dataclasses.FrozenInstanceError):
        runner.max_repair_calls = float("nan")  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        runner.validators = ()  # type: ignore[misc]

    # replace() is the supported way to reconfigure, and it revalidates.
    assert dataclasses.replace(runner, max_repair_calls=3).max_repair_calls == 3
    with pytest.raises(ValueError, match="max_repair_calls"):
        dataclasses.replace(runner, max_repair_calls=float("inf"))  # type: ignore[arg-type]


# --- a repair must not silently change which conversation is being repaired --------------


def test_a_continuation_without_a_new_handle_is_unsatisfied_not_repaired() -> None:
    """The request rode ``previous_turn_handle``, so the conversation lives on the provider's
    side of that handle. A turn that comes back without a new handle leaves nowhere to
    continue from: synthesizing the by-value form would repair against a prompt holding only
    this turn's instruction, discarding everything behind the handle. Stop instead."""

    request = ModelRequest(
        instruction="and now summarize it",
        system_prompt="sys",
        tools=(),
        previous_turn_handle="resp_prior",
    )
    h = _harness(
        [
            ModelTurn(final_text="not json", stop_reason="stop", response_id=None),
            ModelTurn(final_text='{"never": "dispatched"}', stop_reason="stop"),
        ],
        request=request,
    )
    result = h.run()

    assert result.status == "unsatisfied"
    assert len(h.adapter.requests) == 1
    # Budget left over is the observable signal that the shape, not the budget, stopped it.
    assert result.repair_calls_used == 0
    assert result.failures_history != ()


def test_an_observation_only_continuation_is_never_repaired_into_an_empty_prompt() -> None:
    request = ModelRequest(
        instruction=None,
        system_prompt="sys",
        tools=(),
        previous_turn_handle="resp_prior",
    )
    h = _harness([ModelTurn(final_text="not json", stop_reason="stop", response_id=None)])
    h.request = request
    assert h.run().status == "unsatisfied"
    assert len(h.adapter.requests) == 1


@pytest.mark.parametrize("response_id", [None, "resp_1"])
def test_a_one_shot_request_with_no_handle_always_synthesizes_by_value(
    response_id: str | None,
) -> None:
    """The neighbouring shape must keep working, and identically whether or not the provider
    handed back an id: with no incoming handle there is no context behind the request, so the
    synthesized form loses nothing and the answer's id is irrelevant to the choice."""

    h = _harness(
        [
            ModelTurn(final_text="not json", stop_reason="stop", response_id=response_id),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ]
    )
    result = h.run()
    assert result.status == "ok"
    repair = h.adapter.requests[1]
    assert repair.previous_turn_handle is None
    assert repair.messages is not None and len(repair.messages) == 3
    assert repair.messages[0] == {"role": "user", "content": "answer"}


# --- streamed chunks carry the attempt that produced them ---------------------------------


def test_streamed_chunks_are_tagged_with_their_attempt() -> None:
    """A rejected attempt's text is discarded output. Tagging every chunk with its attempt is
    what lets a consumer retract it; without the tag the rejected answer and the corrected one
    arrive as one undelimited stream while the result reports ``status="ok"``."""

    adapter = FakeStreamingModelAdapter(
        chunk_turns=[
            [TextDelta("prose, "), TextDelta("not json")],
            [TextDelta('{"fixed": '), TextDelta("true}")],
        ]
    )
    validator = _Validator()
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=adapter), validators=(validator,), max_repair_calls=1
    )
    seen: list[tuple[int, str]] = []

    async def _drive() -> ValidatedCallResult:
        return await runner.acall(
            ModelRequest(instruction="answer", system_prompt="sys", tools=()),
            delta_consumer=lambda attempt, chunk: seen.append(
                (attempt, getattr(chunk, "text", ""))
            ),
        )

    result = asyncio.run(_drive())

    assert result.status == "ok"
    assert [attempt for attempt, _ in seen] == [0, 0, 1, 1]
    # Everything under attempt 0 is retracted; the answer is exactly the last attempt's text.
    accepted = "".join(text for attempt, text in seen if attempt == result.repair_calls_used)
    assert accepted == result.turn.final_text == '{"fixed": true}'


def test_a_single_attempt_still_streams_under_attempt_zero() -> None:
    adapter = FakeStreamingModelAdapter(chunk_turns=[[TextDelta('{"a": 1}')]])
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=adapter), validators=(_Validator(),)
    )
    seen: list[tuple[int, str]] = []

    async def _drive() -> ValidatedCallResult:
        return await runner.acall(
            ModelRequest(instruction="answer", system_prompt="sys", tools=()),
            delta_consumer=lambda attempt, chunk: seen.append(
                (attempt, getattr(chunk, "text", ""))
            ),
        )

    assert asyncio.run(_drive()).status == "ok"
    assert seen == [(0, '{"a": 1}')]


# --- a tool-call answer is a distinct outcome, not empty text ---------------------------


def _read_tool():
    from monoid_agent_kernel.tools.base import ToolSpec

    return ToolSpec(
        id="fs.read",
        description="read",
        input_schema={"type": "object"},
        capability="fs.read",
        side_effect="read",
        handler=lambda *_a: None,
    )


def test_a_tool_call_turn_short_circuits_before_validation() -> None:
    """A turn that stopped to call tools has no final answer to judge. Validating its empty
    ``final_text`` burned a paid repair on rewriting an answer the model never gave, and the
    repair conversation recorded ``{'role': 'assistant', 'content': ''}`` as what the model
    said. The short-circuit precedes validation, exactly like refusal and truncation."""

    from monoid_agent_kernel.providers.base import ToolCall

    request = ModelRequest(instruction="answer", system_prompt="sys", tools=(_read_tool(),))
    h = _harness(
        [
            ModelTurn(
                final_text=None,
                stop_reason="tool_calls",
                tool_calls=(ToolCall(id="c1", name="fs.read", arguments={}),),
            )
        ],
        request=request,
    )
    result = h.run()

    assert result.status == "tool_calls"
    assert result.turn.tool_calls
    assert h.validator.calls == 0
    assert result.repair_calls_used == 0
    assert len(result.receipts) == 1


def test_a_tool_call_turn_is_not_ok_even_with_no_validators() -> None:
    """The zero-validator path returned ``"ok"`` with ``final_text=None`` for a turn that
    was a tool request — a caller reading ``status == "ok"`` treated "the model asked for a
    tool nobody will run" as a successful answer."""

    from monoid_agent_kernel.providers.base import ToolCall

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                final_text=None,
                stop_reason="tool_calls",
                tool_calls=(ToolCall(id="c1", name="fs.read", arguments={}),),
            )
        ]
    )
    runner = ValidatedCallRunner(runner=ModelCallRunner(adapter=adapter))
    result = asyncio.run(
        runner.acall(
            ModelRequest(instruction="answer", system_prompt="sys", tools=(_read_tool(),))
        )
    )
    assert result.status == "tool_calls"


def test_a_tool_call_turn_without_the_stop_reason_still_short_circuits() -> None:
    """The two signals are one rule: an adapter (a gateway with server-side tools, or a
    misbehaving one) can return tool calls under ``stop_reason="stop"`` — the calls
    themselves are the fact that matters."""

    from monoid_agent_kernel.providers.base import ToolCall

    request = ModelRequest(instruction="answer", system_prompt="sys", tools=())
    h = _harness(
        [
            ModelTurn(
                final_text="",
                stop_reason="stop",
                tool_calls=(ToolCall(id="c1", name="fs.read", arguments={}),),
            )
        ],
        request=request,
    )
    result = h.run()
    assert result.status == "tool_calls"
    assert h.validator.calls == 0


# --- receipts survive an exception ------------------------------------------------------


def test_a_validator_defect_carries_the_receipts_of_every_call_made() -> None:
    """The docstring promises the audit trail is complete whatever the outcome — a defect
    raise that dropped the receipts made both paid calls unaccountable to the caller."""

    class _DefectOnSecond:
        id = "defective"
        schema = None

        def __init__(self) -> None:
            self.calls = 0

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            self.calls += 1
            if self.calls == 1:
                return ValidationOutcome(ok=False, feedback="try again")
            raise RuntimeError("validator defect")

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(final_text="prose", stop_reason="stop"),
            ModelTurn(final_text="still prose", stop_reason="stop"),
        ]
    )
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=adapter),
        validators=(_DefectOnSecond(),),
        max_repair_calls=2,
    )
    with pytest.raises(OutputValidatorError) as defect:
        asyncio.run(runner.acall(ModelRequest(instruction="q", system_prompt="s", tools=())))
    assert len(defect.value.receipts) == 2


def test_receipt_stamping_merges_inner_before_outer() -> None:
    """An adapter that internally delegates to another ValidatedCallRunner propagates an
    exception already carrying the inner call's receipts; the outer stamp appends its own
    completed calls' receipts after them — both trails are paid calls, and either
    overwriting (losing the inner) or skipping (losing the outer) destroys one."""

    from monoid_agent_kernel.validated_call import _stamp_receipts

    error = RuntimeError("boom")
    _stamp_receipts(error, ("inner-receipt",))  # type: ignore[arg-type]
    _stamp_receipts(error, ("outer-receipt",))  # type: ignore[arg-type]
    assert error.receipts == ("inner-receipt", "outer-receipt")  # type: ignore[attr-defined]

    empty_outer = RuntimeError("boom")
    _stamp_receipts(empty_outer, ("inner-receipt",))  # type: ignore[arg-type]
    _stamp_receipts(empty_outer, ())
    assert empty_outer.receipts == ("inner-receipt",)  # type: ignore[attr-defined]


def test_each_validator_sees_its_own_parsed_view() -> None:
    """FinalOutputView is documented read-only. The dataclass is frozen but ``parsed`` was one
    shared mutable object, so validator A's in-place mutation was judged — and surfaced as a
    value — by validator B."""

    class _Mutator:
        id = "mutator"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            if isinstance(view.parsed, dict):
                view.parsed["injected"] = True
            return ValidationOutcome(ok=True, value=view.parsed)

    class _Reader:
        id = "reader"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            return ValidationOutcome(ok=True, value=view.parsed)

    adapter = FakeModelAdapter(turns=[ModelTurn(final_text='{"a": 1}', stop_reason="stop")])
    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=adapter), validators=(_Mutator(), _Reader())
    )
    result = asyncio.run(
        runner.acall(
            ModelRequest(
                instruction="q", system_prompt="s", tools=(), output_schema={"type": "object"}
            )
        )
    )
    assert result.status == "ok"
    values = dict(result.ok_values)
    assert values["reader"] == {"a": 1}
    assert result.value == {"a": 1}


def test_receipts_ride_an_escaping_adapter_error() -> None:
    """A repair call that fails at the boundary loses its own receipt to the adapter (it
    rides ``ModelCallRunner.subscriptions`` only), but the completed calls' receipts must
    not be lost with it."""

    from monoid_agent_kernel.errors import ModelAdapterError

    class _FailsOnSecond:
        def __init__(self) -> None:
            self.requests: list[ModelRequest] = []

        def next_turn(self, request: ModelRequest) -> ModelTurn:
            self.requests.append(request)
            if len(self.requests) == 1:
                return ModelTurn(final_text="prose", stop_reason="stop")
            raise ModelAdapterError("boom", provider_error_code="gateway_network_error")

    runner = ValidatedCallRunner(
        runner=ModelCallRunner(adapter=_FailsOnSecond()), validators=(_Validator(),)
    )
    with pytest.raises(ModelAdapterError) as failed:
        asyncio.run(runner.acall(ModelRequest(instruction="q", system_prompt="s", tools=())))
    assert len(getattr(failed.value, "receipts")) == 1


# --- the repair request carries its conversation exactly one way ------------------------


def test_a_by_value_repair_clears_the_carriage_fields_it_did_not_choose() -> None:
    """The repair follows the shape of how the request carried its conversation — so the
    fields of the shapes it did NOT choose must not ride along. A stale ``instruction`` on a
    messages-shape repair invites a conforming adapter to re-inject the original question,
    and both stale fields dirty the repair's ``request_digest``. Also the both-set
    precedence pin: ``messages`` wins over a handle, exactly as the shipped adapters read
    the original request."""

    request = ModelRequest(
        instruction="latest q",
        system_prompt="sys",
        tools=(),
        messages=({"role": "user", "content": "q"},),
        previous_turn_handle="resp_0",
    )
    h = _harness(
        [
            ModelTurn(final_text="prose", stop_reason="stop", response_id="resp_1"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    assert h.run().status == "ok"
    repair = h.adapter.requests[1]
    assert repair.messages is not None and repair.messages[-2] == {
        "role": "assistant",
        "content": "prose",
    }
    assert repair.instruction is None
    assert repair.previous_turn_handle is None


def test_a_synthesized_repair_clears_the_instruction_it_absorbed() -> None:
    request = ModelRequest(instruction="one shot q", system_prompt="sys", tools=())
    h = _harness(
        [
            ModelTurn(final_text="prose", stop_reason="stop", response_id="resp_1"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    assert h.run().status == "ok"
    repair = h.adapter.requests[1]
    assert repair.messages is not None
    assert repair.messages[0] == {"role": "user", "content": "one shot q"}
    assert repair.instruction is None
    assert repair.previous_turn_handle is None


def test_an_instructionless_one_shot_still_synthesizes_a_consistent_repair() -> None:
    request = ModelRequest(instruction=None, system_prompt="sys", tools=())
    h = _harness(
        [
            ModelTurn(final_text="prose", stop_reason="stop"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ],
        request=request,
    )
    assert h.run().status == "ok"
    repair = h.adapter.requests[1]
    assert repair.messages is not None
    assert repair.messages[0] == {"role": "user", "content": ""}
