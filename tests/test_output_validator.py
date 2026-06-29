"""OutputValidator v1: post-response validation + bounded re-prompt, stop_reason promotion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from conftest import runtime_provider, tool_binding

from native_agent_runner.core.agents import AgentRuntimeConfig, OutputValidatorBinding
from native_agent_runner.core.checkpoint import RunCheckpoint, read_checkpoint, write_checkpoint
from native_agent_runner.core.output_validator import (
    FinalOutputView,
    OutputRetry,
    ValidationOutcome,
)
from native_agent_runner.core.spec import AgentRunSpec, RunLimits
from native_agent_runner.loop import AgentLoop
from native_agent_runner.providers.base import ModelTurn, TextDelta, assemble_streamed_turn
from native_agent_runner.providers.fake import FakeModelAdapter, fake_tool_call
from native_agent_runner.providers.gateway import _parse_gateway_response
from native_agent_runner.providers.openai import _parse_response, _stop_reason_from_response
from native_agent_runner.recorder import MemoryEventSink


# --- validators ---------------------------------------------------------------------------


class StrictJsonValidator:
    """final_text must be a JSON object with a 'summary' key; exercises the OutputRetry
    (bad JSON) and the returned-ValidationOutcome (missing key) rejection paths."""

    id = "json.strict"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        try:
            data = json.loads(view.final_text)
        except json.JSONDecodeError as exc:
            raise OutputRetry(f"not valid JSON: {exc}") from exc
        if "summary" not in data:
            return ValidationOutcome(ok=False, feedback="missing required 'summary' field")
        return ValidationOutcome(ok=True, value=data)


class ContainsOkValidator:
    id = "contains.ok"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        if "ok" in view.final_text:
            return ValidationOutcome(ok=True, value=view.final_text)
        return ValidationOutcome(ok=False, feedback="final answer must contain 'ok'")


class DefectValidator:
    id = "defect"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        raise KeyError("a bug in the validator")  # not OutputRetry/ValueError → defect


class MalformedReturnValidator:
    """Returns None instead of a ValidationOutcome — a validator bug. Must be classified as a
    defect (output_validator_error) with the validator id, not an uncaught AttributeError."""

    id = "malformed"
    schema = None

    def validate(self, view: FinalOutputView):  # type: ignore[override]
        return None  # noqa: RET501 - intentional bug under test


class RequireFoo:
    id = "require.foo"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        if "FOO" in view.final_text:
            return ValidationOutcome(ok=True, value="foo")
        return ValidationOutcome(ok=False, feedback="must contain FOO")


class ForbidFoo:
    id = "forbid.foo"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        if "FOO" not in view.final_text:
            return ValidationOutcome(ok=True, value="no-foo")
        return ValidationOutcome(ok=False, feedback="must NOT contain FOO")


class SlowValidator:
    """Blocks briefly — exercises the asyncio.to_thread offload (E4) end to end."""

    id = "slow"
    schema = None

    def validate(self, view: FinalOutputView) -> ValidationOutcome:
        import time

        time.sleep(0.05)
        return ValidationOutcome(ok=True, value="slow-ok")


# --- harness ------------------------------------------------------------------------------


def _spec(tmp_path: Path, *, limits: RunLimits | None = None) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=limits or RunLimits(),
    )


def _provider(*validator_ids: str, tools: tuple[str, ...] = ()):
    config = AgentRuntimeConfig(
        definition_id="test-agent",
        tools=tuple(tool_binding(t) for t in tools),
        output_validators=tuple(
            OutputValidatorBinding(validator_id=v, enabled=True) for v in validator_ids
        ),
    )
    return runtime_provider(config)


def _text_turn(text: str, *, stop_reason: str | None = "stop") -> ModelTurn:
    return ModelTurn(response_id="r", final_text=text, stop_reason=stop_reason)


# --- validation happy / retry / exhaustion ------------------------------------------------


def test_validator_accepts_valid_json_and_sets_final_output(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "all good"}')])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == {"summary": "all good"}


def test_validator_retries_then_succeeds(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn("not json at all"), _text_turn('{"summary": "fixed"}')])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == {"summary": "fixed"}
    assert len(adapter.requests) == 2  # original + one repair re-prompt


def test_validator_exhaustion_settles_limited(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn("bad one"), _text_turn("bad two")])
    result = AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_output_retries=1)),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "limited"
    assert result.error_code == "output_validator_unsatisfied"


def test_validator_defect_terminalizes(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn("anything")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("defect"),
        output_validators=(DefectValidator(),),
    ).run_once("go")

    assert result.status == "failed"
    assert result.error_code == "output_validator_error"


def test_malformed_validator_return_terminalizes_as_defect(tmp_path: Path) -> None:
    # A validator that returns None (no ok/feedback) is a bug the model cannot fix; the AttributeError
    # it would otherwise trigger downstream must be classified as output_validator_error, not a
    # generic internal error.
    adapter = FakeModelAdapter(turns=[_text_turn("anything")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("malformed"),
        output_validators=(MalformedReturnValidator(),),
    ).run_once("go")

    assert result.status == "failed"
    assert result.error_code == "output_validator_error"


# --- textless refusal / truncation (review fix ①) -----------------------------------------


def test_textless_refusal_settles_output_refused(tmp_path: Path) -> None:
    # An OpenAI refusal content part yields stop_reason="refusal" with NO final text — it must
    # still settle output_refused, not the "neither text nor tool calls" model error.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=None, stop_reason="refusal")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "failed"
    assert result.error_code == "output_refused"


def test_textless_truncation_settles_output_truncated(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=None, stop_reason="length")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "limited"
    assert result.error_code == "output_truncated"


# --- per-turn validator gating honors a config hot-swap (review fix ②) ---------------------


def test_active_validators_resolve_from_per_turn_config(tmp_path: Path) -> None:
    # Gating resolves from the *given* config, so a mid-run replace_runtime_config that adds a
    # disabling binding takes effect (vs the old bootstrap-frozen tuple).
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=FakeModelAdapter(),
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    )
    enabled = AgentRuntimeConfig(definition_id="t", tools=())  # default-on, no binding
    disabled = AgentRuntimeConfig(
        definition_id="t",
        tools=(),
        output_validators=(OutputValidatorBinding(validator_id="json.strict", enabled=False),),
    )
    assert len(loop._active_output_validators(enabled)) == 1
    assert loop._active_output_validators(disabled) == ()


# --- refusal / truncation (item A) --------------------------------------------------------


def test_refusal_settles_output_refused_without_validating(tmp_path: Path) -> None:
    # A refusal must NOT be validated and must not settle as completed.
    adapter = FakeModelAdapter(turns=[_text_turn("I won't", stop_reason="refusal")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "failed"
    assert result.error_code == "output_refused"


def test_truncation_settles_output_truncated_without_validating(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "complete enough"}', stop_reason="length")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "limited"
    assert result.error_code == "output_truncated"


# --- run.finish settle path (item B: per-site cleanup, no infinite re-settle) --------------


def test_run_finish_path_validates_and_repairs(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("run_finish", {"summary": "bad"}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("run_finish", {"summary": "ok now"}, "c2"),)),
        ]
    )
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("run.finish",)),
        output_validators=(ContainsOkValidator(),),
    ).run_once("go")

    assert result.status == "completed"  # cleared context.finished → no infinite re-settle
    assert result.final_output == "ok now"
    assert len(adapter.requests) == 2


# --- gating: default ON, binding is an opt-out --------------------------------------------


def test_registered_validator_runs_by_default_without_binding(tmp_path: Path) -> None:
    # Default-on: a registered validator runs even when the config has no binding for it.
    config = AgentRuntimeConfig(definition_id="test-agent", tools=(), output_validators=())
    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "ran by default"}')])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == {"summary": "ran by default"}


def test_disabled_binding_skips_validator(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    # An enabled=False binding is the per-run opt-out → the validator must not run.
    config = AgentRuntimeConfig(
        definition_id="test-agent",
        tools=(),
        output_validators=(OutputValidatorBinding(validator_id="json.strict", enabled=False),),
    )
    adapter = FakeModelAdapter(turns=[_text_turn("plain prose, not json")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        output_validators=(StrictJsonValidator(),),
        event_sinks=(sink,),
    ).run_once("go")

    assert result.status == "completed"  # disabled → never ran, so the prose is accepted
    skipped = [e for e in sink.events if e.type == "output.validator.skipped"]
    assert any(
        e.data.get("validator_id") == "json.strict" and e.data.get("reason") == "disabled"
        for e in skipped
    )


def test_from_tools_enables_validator_one_liner(tmp_path: Path) -> None:
    # DX payoff of default-on: the convenience constructor activates a validator with no binding.
    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "via from_tools"}')])
    result = AgentLoop.from_tools(
        _spec(tmp_path),
        adapter,
        [],
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == {"summary": "via from_tools"}


# --- checkpoint round-trip (item D) -------------------------------------------------------


def test_output_retries_survives_checkpoint_round_trip(tmp_path: Path) -> None:
    cp = RunCheckpoint(run_id="run_1", output_retries=2)
    write_checkpoint(tmp_path, cp)
    restored = read_checkpoint(tmp_path)
    assert restored is not None
    assert restored.output_retries == 2  # else a mid-repair restart double-grants the budget


# --- item A: stop_reason promotion across adapters -----------------------------------------


def test_openai_stop_reason_mapping() -> None:
    assert _stop_reason_from_response({"status": "completed"}, tool_calls_present=False) == "stop"
    assert _stop_reason_from_response({"status": "completed"}, tool_calls_present=True) == "tool_calls"
    assert (
        _stop_reason_from_response(
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            tool_calls_present=False,
        )
        == "length"
    )
    assert (
        _stop_reason_from_response(
            {"status": "incomplete", "incomplete_details": {"reason": "content_filter"}},
            tool_calls_present=False,
        )
        == "refusal"
    )
    # A refusal content part on an otherwise-complete response.
    refusal_doc = {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal"}]}]}
    assert _stop_reason_from_response(refusal_doc, tool_calls_present=False) == "refusal"


def test_openai_parse_response_carries_stop_reason() -> None:
    turn = _parse_response({"id": "x", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]})
    assert turn.stop_reason == "stop"


def test_fake_streaming_infers_stop_reason() -> None:
    turn = assemble_streamed_turn([TextDelta("hello")])
    assert turn.stop_reason == "stop"


def test_gateway_wire_round_trips_stop_reason() -> None:
    turn = _parse_gateway_response({"final_text": "done", "tool_calls": [], "stop_reason": "length"})
    assert turn.stop_reason == "length"
    # Older gateway without the field: inferred.
    inferred = _parse_gateway_response({"final_text": "done", "tool_calls": []})
    assert inferred.stop_reason == "stop"


# --- E4: validation offloaded to a thread (slow validator still settles) ------------------


def test_slow_validator_still_settles(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn("anything")])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("slow"),
        output_validators=(SlowValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == "slow-ok"


# --- E2: contradictory validators exhaust with a diagnosable roll-up -----------------------


def test_contradictory_validators_exhaust_with_failure_rollup(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    # No text can satisfy both (require FOO and forbid FOO) → deterministic exhaustion.
    adapter = FakeModelAdapter(turns=[_text_turn("hello"), _text_turn("FOO here")])
    result = AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_output_retries=1)),
        model_adapter=adapter,
        runtime_config_provider=_provider("require.foo", "forbid.foo"),
        output_validators=(RequireFoo(), ForbidFoo()),
        event_sinks=(sink,),
    ).run_once("go")

    assert result.status == "limited"
    assert result.error_code == "output_validator_unsatisfied"

    exhausted = [e for e in sink.events if e.type == "output.validator.exhausted"]
    assert exhausted
    by_validator = exhausted[-1].data["failures_by_validator"]
    # Both validators show up as failing across attempts — the contradiction signal.
    assert by_validator.get("require.foo") and by_validator.get("forbid.foo")
    # Same roll-up surfaced in the run result metrics.
    assert set(result.metrics["output_validation"]["failures_by_validator"]) == {"require.foo", "forbid.foo"}


# --- E3: per-validator outputs keyed by id -------------------------------------------------


def test_outputs_keyed_by_validator_id(tmp_path: Path) -> None:
    class JsonValue:
        id = "json.v"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            return ValidationOutcome(ok=True, value=json.loads(view.final_text))

    class LenValue:
        id = "len.v"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            return ValidationOutcome(ok=True, value=len(view.final_text))

    text = '{"x": 1}'
    adapter = FakeModelAdapter(turns=[_text_turn(text)])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.v", "len.v"),
        output_validators=(JsonValue(), LenValue()),
    ).run_once("go")

    assert result.status == "completed"
    assert result.outputs["json.v"] == {"x": 1}
    assert result.outputs["len.v"] == len(text)
    assert result.final_output == len(text)  # last ok wins (registration order)


# --- F3: item B — a repair turn calling a NON-finish tool must not re-settle on a stale flag


def test_repair_turn_with_non_finish_tool_does_not_resettle(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("run_finish", {"summary": "bad"}, "c1"),)),
            ModelTurn(response_id="r2", tool_calls=(fake_tool_call("fs_write", {"path": "note.md", "content": "written"}, "c2"),)),
            ModelTurn(response_id="r3", tool_calls=(fake_tool_call("run_finish", {"summary": "ok done"}, "c3"),)),
        ]
    )
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("fs.write", "run.finish")),
        output_validators=(ContainsOkValidator(),),
    ).run_once("go")

    # Reaching the third turn's "ok done" (not settling on turn 1's "bad" summary) proves
    # context.finished was cleared: without item B, the fs.write turn would re-settle on the stale
    # flag and the run would exhaust as `limited` at turn 2, never reaching turn 3.
    assert result.status == "completed"
    assert result.final_output == "ok done"
    assert len(adapter.requests) == 3  # all three turns ran; no premature re-settle


# --- run.finish repair: don't leak rejected-finish metadata / drop its tool output (re-review) -


def test_rejected_finish_metadata_cleared_on_natural_repair(tmp_path: Path) -> None:
    # A rejected run.finish (with outputs) repaired by a plain final-text answer must NOT surface
    # the rejected finish's outputs at close().
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "bad", "outputs": ["stale.txt"]}, "c1"),),
            ),
            _text_turn("ok now"),
        ]
    )
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("run.finish",)),
        output_validators=(ContainsOkValidator(),),
    ).run_once("go")

    assert result.status == "completed"
    assert result.final_output == "ok now"
    assert result.final_outputs == ()  # the rejected finish's "stale.txt" must not leak


def test_rejected_finish_tool_output_preserved_in_messages(tmp_path: Path) -> None:
    # The repair turn must carry the run.finish function_call_output (tool message) BEFORE the
    # validator-feedback user message — else a by-value adapter sees a dangling function_call.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("run_finish", {"summary": "bad"}, "c1"),)),
            _text_turn("ok now"),
        ]
    )
    AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("run.finish",)),
        output_validators=(ContainsOkValidator(),),
    ).run_once("go")

    repair_msgs = adapter.requests[1].messages or ()
    tool_idx = next(
        i for i, m in enumerate(repair_msgs) if m.get("role") == "tool" and m.get("call_id") == "c1"
    )
    user_repair_idx = next(
        i for i, m in enumerate(repair_msgs)
        if m.get("role") == "user" and "did not satisfy" in str(m.get("content", ""))
    )
    assert tool_idx < user_repair_idx


def test_exhausted_rejected_finish_metadata_cleared(tmp_path: Path) -> None:
    # A run.finish rejected to exhaustion (max_output_retries=0) must not surface its outputs.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "bad", "outputs": ["stale.txt"]}, "c1"),),
            ),
        ]
    )
    result = AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_output_retries=0)),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("run.finish",)),
        output_validators=(ContainsOkValidator(),),
    ).run_once("go")

    assert result.status == "limited"
    assert result.error_code == "output_validator_unsatisfied"
    assert result.final_outputs == ()  # rejected finish's "stale.txt" must not leak


def test_unknown_opt_out_binding_warns_at_bootstrap(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    # A misspelled disable-binding ("json.stirct") matches no registered validator → it must be
    # surfaced (not silently ignored), and the real validator ("json.strict") still runs.
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(),
        output_validators=(OutputValidatorBinding(validator_id="json.stirct", enabled=False),),
    )
    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "ok"}')])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        output_validators=(StrictJsonValidator(),),
        event_sinks=(sink,),
    ).run_once("go")

    assert result.status == "completed"  # the real validator still ran (and passed)
    skipped = [e for e in sink.events if e.type == "output.validator.skipped"]
    assert any(
        e.data.get("validator_id") == "json.stirct" and e.data.get("reason") == "unknown_binding"
        for e in skipped
    )


def test_finish_output_logged_when_all_validators_disabled(tmp_path: Path) -> None:
    # Validators registered but all disabled → a run.finish still logs its tool output before the
    # run parks (the no-active-validator fast path must not skip the finish bookkeeping).
    config = AgentRuntimeConfig(
        definition_id="t",
        tools=(tool_binding("run.finish"),),
        output_validators=(OutputValidatorBinding(validator_id="contains.ok", enabled=False),),
    )
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("run_finish", {"summary": "anything"}, "c1"),)),
            _text_turn("second answer"),
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(config),
        output_validators=(ContainsOkValidator(),),
    )
    loop.open()
    loop.submit("first")
    loop.submit("second")
    loop.close()

    second_msgs = adapter.requests[1].messages or ()
    tool_idx = next(
        i for i, m in enumerate(second_msgs) if m.get("role") == "tool" and m.get("call_id") == "c1"
    )
    user2_idx = next(
        i for i, m in enumerate(second_msgs) if m.get("role") == "user" and "second" in str(m.get("content", ""))
    )
    assert tool_idx < user2_idx


def test_prior_finish_outputs_not_leaked_into_next_natural_turn(tmp_path: Path) -> None:
    # Turn A settles via run.finish (with outputs); turn B settles naturally. Turn B's validator
    # view and result must not carry turn A's finish outputs.
    seen: list = []

    class _CaptureOutputs:
        id = "capture"
        schema = None

        def validate(self, view: FinalOutputView) -> ValidationOutcome:
            seen.append(view.final_outputs)
            return ValidationOutcome(ok=True, value=None)

    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done", "outputs": ["a.txt"]}, "c1"),),
            ),
            _text_turn("plain answer"),
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("capture", tools=("run.finish",)),
        output_validators=(_CaptureOutputs(),),
    )
    loop.open()
    loop.submit("first")
    loop.submit("second")
    run_result = loop.close()

    assert seen[0] == ("a.txt",)  # turn A's finish exposed its own outputs...
    assert seen[-1] == ()  # ...but turn B (natural) saw none — no stale leak
    assert run_result.final_outputs == ()


def test_successful_finish_tool_output_logged_before_next_user_message(tmp_path: Path) -> None:
    # A validated run.finish in a multi-turn session must log its function_call_output before the
    # run parks, or the next user message interleaves ahead of it (dangling function_call).
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", tool_calls=(fake_tool_call("run_finish", {"summary": "ok done"}, "c1"),)),
            _text_turn("ok bye"),
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("contains.ok", tools=("run.finish",)),
        output_validators=(ContainsOkValidator(),),
    )
    loop.open()
    loop.submit("first")
    loop.submit("second")
    loop.close()

    second_msgs = adapter.requests[1].messages or ()
    tool_idx = next(
        i for i, m in enumerate(second_msgs) if m.get("role") == "tool" and m.get("call_id") == "c1"
    )
    user2_idx = next(
        i for i, m in enumerate(second_msgs) if m.get("role") == "user" and "second" in str(m.get("content", ""))
    )
    assert tool_idx < user2_idx


# --- output_retries counts re-prompts, not failed attempts (review fix ⑤) -----------------


def test_output_retries_counts_reprompts_not_failed_attempts(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    adapter = FakeModelAdapter(turns=[_text_turn("bad one"), _text_turn("bad two")])
    result = AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_output_retries=1)),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
        event_sinks=(sink,),
    ).run_once("go")

    assert result.status == "limited"
    exhausted = [e for e in sink.events if e.type == "output.validator.exhausted"][-1]
    assert exhausted.data["retries"] == 1  # exactly one re-prompt issued...
    assert len(exhausted.data["history"]) == 2  # ...though two attempts failed
    assert result.metrics["output_validation"]["retries"] == 1


def test_zero_retries_exhausts_with_zero_count(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[_text_turn("bad")])
    result = AgentLoop(
        spec=_spec(tmp_path, limits=RunLimits(max_output_retries=0)),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.status == "limited"
    assert result.metrics["output_validation"]["retries"] == 0  # no re-prompt occurred
    assert len(adapter.requests) == 1  # single attempt, no re-prompt turn


# --- turn.settled validation summary ------------------------------------------------------


def test_turn_settled_carries_validation_summary(tmp_path: Path) -> None:
    sink = MemoryEventSink()
    adapter = FakeModelAdapter(turns=[_text_turn("not json"), _text_turn('{"summary": "ok"}')])
    AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
        event_sinks=(sink,),
    ).run_once("go")

    settled = [e for e in sink.events if e.type == "turn.settled"]
    assert settled
    assert settled[-1].data["output_validators"] == 1
    assert settled[-1].data["output_retries"] == 1  # one re-prompt before the good answer


# --- output_as typed accessor (Q1) --------------------------------------------------------


def test_output_as_coerces_to_typed_value(tmp_path: Path) -> None:
    @dataclass
    class Answer:
        summary: str

    adapter = FakeModelAdapter(turns=[_text_turn('{"summary": "done"}')])
    result = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=_provider("json.strict"),
        output_validators=(StrictJsonValidator(),),
    ).run_once("go")

    assert result.final_output == {"summary": "done"}  # untyped object
    typed = result.output_as(Answer)  # restored static type
    assert isinstance(typed, Answer)
    assert typed.summary == "done"


# --- validate() flags an unknown validator binding (Q2) -----------------------------------


def test_validate_flags_unknown_validator_binding() -> None:
    bad = AgentRuntimeConfig(
        definition_id="t",
        tools=(),
        output_validators=(OutputValidatorBinding(validator_id="typo.id", enabled=False),),
    )
    issues = AgentLoop.validate(bad, output_validators=(StrictJsonValidator(),))
    assert any("typo.id" in issue for issue in issues)

    good = AgentRuntimeConfig(
        definition_id="t",
        tools=(),
        output_validators=(OutputValidatorBinding(validator_id="json.strict", enabled=False),),
    )
    assert not any("validator_id" in issue for issue in AgentLoop.validate(good, output_validators=(StrictJsonValidator(),)))
