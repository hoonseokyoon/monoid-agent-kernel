"""W5 PR 3: the standalone validated model call.

The boundary pins here are the contract (v0.20 scope doc §4-3): a repair call NEVER carries
tools, and refusal/truncation short-circuit BEFORE any validator runs. The repair text must be
byte-identical to the loop's -- both import ``build_repair_message`` from
``core.output_validator``, and the parity test is the observable proof of that sharing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from monoid_agent_kernel.core.output_validator import (
    FinalOutputView,
    OutputValidatorError,
    ValidationOutcome,
    build_repair_message,
)
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter
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


def test_repair_uses_the_provider_handle_when_present() -> None:
    h = _harness(
        [
            ModelTurn(final_text="nope", stop_reason="stop", response_id="resp_1"),
            ModelTurn(final_text='{"fixed": true}', stop_reason="stop"),
        ]
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


def test_negative_budget_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="max_repair_calls"):
        ValidatedCallRunner(
            runner=ModelCallRunner(adapter=FakeModelAdapter()), max_repair_calls=-1
        )
