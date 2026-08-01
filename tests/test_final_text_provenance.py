"""Provenance of ``RunState.final_text`` — who wrote the text a run settles on.

Only model-authored text is content the fan-out event stream must not carry. Kernel-authored
limit and cancellation strings are ours to publish, and digesting them would cost an operator the
one sentence explaining why the run stopped.

Every rule here is pinned from **both** sides. A test that only proves the flag goes up cannot
tell a correct implementation from one stuck at ``True``, and the two failure directions are not
symmetric: stuck ``False`` publishes model output on a stream that is documented as redacted,
while stuck ``True`` merely hides a limit message. Both are defects; only one is a disclosure.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import RunCheckpoint
from monoid_agent_kernel.core.content import ImagePart, TextPart
from monoid_agent_kernel.core.spec import AgentRunSpec, RunLimits
from monoid_agent_kernel.loop import AgentLoop, RunState
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import (
    FakeModelAdapter,
    FakeMultimodalModelAdapter,
    fake_tool_call,
)

# Every module under src/, not just loop.py. Scoping the guard to the file the rule was written
# in is what let two assignments in loop_phases.py ship unpaired: proving a rule on one of two
# parallel halves and never binding the twin is the defect shape this repo keeps re-earning.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel"
_ASSIGNMENT = re.compile(r"\s*state\.final_text = ")
_PROVENANCE = "state.final_text_is_model_output = "
_PNG_BYTES = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"


def _watch(loop: AgentLoop) -> list[tuple[str, str, bool]]:
    """Record ``(event, final_text, is_model_output)`` at the two points PR 14 branches on.

    ``_finalize`` feeds ``run.finished`` and ``_checkpoint_on_settle`` feeds ``turn.settled`` —
    the two payloads that carry model text today. Reading the field anywhere else would prove
    nothing about the events that actually leak. Both are invoked as ``self._method(...)``, so
    shadowing them on the instance intercepts every call without subclassing a dataclass.
    """
    seen: list[tuple[str, str, bool]] = []
    original_settle = loop._checkpoint_on_settle
    original_finalize = loop._finalize

    def settle(state: RunState, res: Any) -> Any:
        seen.append(("turn.settled", state.final_text, state.final_text_is_model_output))
        return original_settle(state, res)

    def finalize(state: RunState, res: Any) -> Any:
        seen.append(("run.finished", state.final_text, state.final_text_is_model_output))
        return original_finalize(state, res)

    loop._checkpoint_on_settle = settle  # type: ignore[method-assign]
    loop._finalize = finalize  # type: ignore[method-assign]
    return seen


def _finished(seen: list[tuple[str, str, bool]]) -> tuple[str, bool]:
    """The ``run.finished`` observation — every closed run produces exactly one."""
    finals = [(text, flagged) for event, text, flagged in seen if event == "run.finished"]
    assert len(finals) == 1, f"expected one run.finished observation, got {seen}"
    return finals[0]


def _spec(tmp_path: Path, **limits: Any) -> AgentRunSpec:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(**limits) if limits else RunLimits(),
    )


def _run(spec: AgentRunSpec, adapter: Any, *tool_ids: str, **loop_kwargs: Any) -> tuple[str, bool]:
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("run.finish",)))),
        **loop_kwargs,
    )
    seen = _watch(loop)
    loop.run_once("go")
    return _finished(seen)


# --- model-authored: the two sites whose text must leave the event stream -------------------


def test_model_response_text_is_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:2964 — the model's own response text.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="the model wrote this")])

    text, flagged = _run(_spec(tmp_path), adapter)

    assert text == "the model wrote this"
    assert flagged is True


def test_refusal_with_no_text_is_still_the_models_turn(tmp_path: Path) -> None:
    # Same site, empty value. Provenance describes who produced the value, not whether it has
    # anything in it — a refusal is the model speaking even when it says nothing. ``stop_reason``
    # is what routes this to the settle branch; putting it in ``raw`` instead sends the turn to
    # "neither text nor tool calls" and tests nothing.
    adapter = FakeModelAdapter(
        turns=[ModelTurn(response_id="r1", final_text=None, stop_reason="refusal")]
    )

    text, flagged = _run(_spec(tmp_path), adapter)

    assert text == ""
    assert flagged is True


def test_run_finish_summary_is_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:3010 — ``summary`` is an argument the model passed to the run.finish tool, so it is
    # model prose even though the kernel stores it. This is the site most easily misread as ours.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "I tidied the notes."}, "c1"),),
            )
        ]
    )

    text, flagged = _run(_spec(tmp_path), adapter)

    assert text == "I tidied the notes."
    assert flagged is True


# --- kernel-authored: text that must stay inline --------------------------------------------


def _tool_calling_adapter(count: int = 6) -> FakeModelAdapter:
    """An adapter that only ever asks for tools, so step/tool-call limits are what stop it."""
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id=f"r{index}",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, f"c{index}"),),
            )
            for index in range(count)
        ]
    )


@pytest.mark.parametrize(
    ("limits", "expected_text"),
    [
        pytest.param(
            {"max_message_log_bytes": 10},
            "Stopped after reaching the conversation size limit.",
            id="conversation-size",
        ),
        pytest.param(
            {"max_steps": 1},
            "Stopped after reaching max steps.",
            id="max-steps",
        ),
    ],
)
def test_kernel_limit_text_is_not_flagged_as_model_output(
    tmp_path: Path, limits: dict[str, Any], expected_text: str
) -> None:
    text, flagged = _run(_spec(tmp_path, **limits), _tool_calling_adapter(), "fs.list", "run.finish")

    assert text == expected_text
    assert flagged is False


def test_max_tool_calls_text_is_not_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:2981.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_list", {"path": "."}, "c1"),
                    fake_tool_call("fs_list", {"path": "."}, "c2"),
                ),
            )
        ]
    )

    text, flagged = _run(_spec(tmp_path, max_tool_calls=1), adapter, "fs.list", "run.finish")

    assert text == "Stopped after reaching max tool calls."
    assert flagged is False


def test_workspace_delta_limit_text_is_not_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:2787.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_write", {"path": "big.txt", "content": "x" * 50}, "c1"),
                ),
            ),
            ModelTurn(
                response_id="r2",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c2"),),
            ),
        ]
    )

    text, flagged = _run(
        _spec(tmp_path, max_delta_file_bytes=10), adapter, "fs.write", "run.finish"
    )

    assert text == "Stopped after reaching the workspace change size limit."
    assert flagged is False


def test_token_budget_text_is_not_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:2775 — checked at the start of turn 2 against turn 1's reported usage.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),),
                usage={"input_tokens": 90, "output_tokens": 20, "total_tokens": 110},
            ),
            ModelTurn(response_id="r2", final_text="never reached"),
        ]
    )

    text, flagged = _run(
        _spec(tmp_path, max_total_tokens=10), adapter, "fs.list", "run.finish"
    )

    assert text == "Stopped after reaching the token budget."
    assert flagged is False


def test_oversize_wire_limit_text_is_not_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:2828 — the resolved (base64) payload trips the cap while the by-reference log is
    # tiny, which is the only way to reach this site rather than the conversation-size one above.
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    workspace.joinpath("big.png").write_bytes(_PNG_BYTES + b"\x00" * 5000)
    adapter = FakeMultimodalModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c1"),),
            )
        ]
    )
    spec = AgentRunSpec(
        workspace_root=workspace,
        run_root=tmp_path / "runs",
        limits=RunLimits(max_message_log_bytes=2000),
    )
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    seen = _watch(loop)
    loop.run_once((TextPart("describe"), ImagePart(source_ref="big.png", mime_type="image/png")))

    text, flagged = _finished(seen)
    assert text == "Stopped after reaching the model request size limit."
    assert flagged is False


def test_cancelled_run_text_is_not_flagged_as_model_output(tmp_path: Path) -> None:
    # loop.py:1221.
    token = CancellationToken()
    token.cancel()
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="never reached")])

    text, flagged = _run(_spec(tmp_path), adapter, "run.finish", cancellation_token=token)

    assert text == "Stopped because the run was cancelled."
    assert flagged is False


def test_failed_run_clears_the_flag(tmp_path: Path) -> None:
    # loop.py:1663 — a terminal failure wipes final_text; the flag must come down with it, or a
    # failed run reports "no text" while still claiming the text was the model's.
    # Neither text nor tool calls -> ModelAdapterError -> terminal failure. (An adapter with an
    # empty ``turns`` list does *not* fail; it hands back a default "fake model completed" turn.)
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1")])

    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    seen = _watch(loop)
    loop.run_once("go")

    text, flagged = _finished(seen)
    assert text == ""
    assert flagged is False


# --- the counterweight: provenance must come *down*, not just up ----------------------------


def test_a_limit_reached_with_provenance_already_up_brings_it_down(tmp_path: Path) -> None:
    """A limit can be reached in a submit whose flag is *already* up, and must lower it.

    The reachable path is the output-validator repair loop: the model settles (flag up at
    loop.py:2964), a validator rejects the text, the loop ``continue``s and re-pumps
    (loop.py:2970-2971), and a later step trips a limit whose text is the kernel's. The
    per-submit reset at loop.py:1195 ran long before that, so only the limit site's *own* reset
    can bring the flag down.

    Note what this test is *not*: driving it through a real validator needs validator-registry
    wiring, and going through a second ``submit()`` proves nothing — that path is covered by the
    per-submit reset and passes even when the limit sites never reset at all (verified: that
    version of this test survived a mutant which neutered the limit-site reset). So the flag is
    raised at the top of the pump instead, which is precisely the state the repair loop leaves
    behind.
    """
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(
                    fake_tool_call("fs_list", {"path": "."}, "c1"),
                    fake_tool_call("fs_list", {"path": "."}, "c2"),
                ),
            )
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path, max_tool_calls=1),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.list", "run.finish")),
    )
    seen = _watch(loop)
    original_pump = loop._apump_turn

    async def pump(state: RunState, res: Any, session: Any) -> Any:
        state.final_text_is_model_output = True  # as a rejected settle would have left it
        return await original_pump(state, res, session)

    loop._apump_turn = pump  # type: ignore[method-assign]
    loop.run_once("go")

    text, flagged = _finished(seen)
    assert text == "Stopped after reaching max tool calls."
    assert flagged is False


def _run_with_provenance_already_up(
    spec: AgentRunSpec, adapter: Any, *tool_ids: str, **loop_kwargs: Any
) -> tuple[str, bool]:
    """Drive a run whose flag is raised at the top of every pump.

    That is the state the output-validator repair loop leaves behind — `loop.py:2970` `continue`s
    and re-pumps after a settle raised the flag — and a validator *defect* raised from
    `LoopSettleCoordinator.apply` reaches `_record_failure` the same way. The per-submit reset at
    `loop.py:1208` has long since run, so only the site's OWN reset can bring the flag down.
    """
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config(*(tool_ids or ("run.finish",)))),
        **loop_kwargs,
    )
    seen = _watch(loop)
    original_pump = loop._apump_turn

    async def pump(state: RunState, res: Any, session: Any) -> Any:
        state.final_text_is_model_output = True
        return await original_pump(state, res, session)

    loop._apump_turn = pump  # type: ignore[method-assign]
    loop.run_once("go")
    return _finished(seen)


def _limit_scenarios() -> list[Any]:
    """Every kernel-authored settle site, with the trigger that reaches it."""
    return [
        pytest.param(
            {"max_steps": 1},
            lambda: _tool_calling_adapter(),
            ("fs.list", "run.finish"),
            {},
            "Stopped after reaching max steps.",
            id="max-steps",
        ),
        pytest.param(
            {"max_message_log_bytes": 10},
            lambda: _tool_calling_adapter(),
            ("fs.list", "run.finish"),
            {},
            "Stopped after reaching the conversation size limit.",
            id="conversation-size",
        ),
        pytest.param(
            {"max_tool_calls": 1},
            lambda: FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="r1",
                        tool_calls=(
                            fake_tool_call("fs_list", {"path": "."}, "c1"),
                            fake_tool_call("fs_list", {"path": "."}, "c2"),
                        ),
                    )
                ]
            ),
            ("fs.list", "run.finish"),
            {},
            "Stopped after reaching max tool calls.",
            id="max-tool-calls",
        ),
        pytest.param(
            {"max_delta_file_bytes": 10},
            lambda: FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="r1",
                        tool_calls=(
                            fake_tool_call(
                                "fs_write", {"path": "big.txt", "content": "x" * 50}, "c1"
                            ),
                        ),
                    ),
                    ModelTurn(
                        response_id="r2",
                        tool_calls=(fake_tool_call("run_finish", {"summary": "done"}, "c2"),),
                    ),
                ]
            ),
            ("fs.write", "run.finish"),
            {},
            "Stopped after reaching the workspace change size limit.",
            id="workspace-delta",
        ),
        pytest.param(
            {"max_total_tokens": 10},
            lambda: FakeModelAdapter(
                turns=[
                    ModelTurn(
                        response_id="r1",
                        tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),),
                        usage={"input_tokens": 90, "output_tokens": 20, "total_tokens": 110},
                    ),
                    ModelTurn(response_id="r2", final_text="never reached"),
                ]
            ),
            ("fs.list", "run.finish"),
            {},
            "Stopped after reaching the token budget.",
            id="token-budget",
        ),
    ]


@pytest.mark.parametrize(("limits", "adapter_factory", "tools", "kwargs", "expected"), _limit_scenarios())
def test_every_kernel_site_lowers_provenance_that_was_already_up(
    tmp_path: Path,
    limits: dict[str, Any],
    adapter_factory: Any,
    tools: tuple[str, ...],
    kwargs: dict[str, Any],
    expected: str,
) -> None:
    """Generalises the max-tool-calls counterweight to every kernel settle site.

    The plain kernel tests above enter each site with the flag already `False` — the per-submit
    reset ran first — so the site's own reset is a no-op *in those tests* and neutering it to a
    self-assignment leaves them green. Verified: seven of the ten reset sites survived exactly
    that mutant while only max-tool-calls, which had this counterweight, died.
    """
    text, flagged = _run_with_provenance_already_up(
        _spec(tmp_path, **limits), adapter_factory(), *tools, **kwargs
    )

    assert text == expected
    assert flagged is False


def test_a_cancelled_run_lowers_provenance_that_was_already_up(tmp_path: Path) -> None:
    token = CancellationToken()
    token.cancel()
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="never reached")])

    text, flagged = _run_with_provenance_already_up(
        _spec(tmp_path), adapter, "run.finish", cancellation_token=token
    )

    assert text == "Stopped because the run was cancelled."
    assert flagged is False


def test_a_terminal_failure_lowers_provenance_that_was_already_up(tmp_path: Path) -> None:
    # Reachable for real: a validator *defect* (any non-ValueError) raised from the settle
    # coordinator reaches `_record_failure` with the flag up and model prose in `final_text`.
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1")])

    text, flagged = _run_with_provenance_already_up(_spec(tmp_path), adapter)

    assert text == ""
    assert flagged is False


def test_a_fresh_submit_resets_provenance(tmp_path: Path) -> None:
    # loop.py:1195 — the per-submit reset block.
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(response_id="r1", final_text="a real answer"),
            ModelTurn(response_id="r2", final_text="another answer"),
        ]
    )
    loop = AgentLoop(
        spec=_spec(tmp_path),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop.open()
    loop.submit("first")
    assert loop._session is not None
    assert loop._session.state.final_text_is_model_output is True

    # Reaching into the reset itself: after a fresh submit begins, the flag is down until this
    # turn's own settle raises it. Asserting only the post-settle value would pass even if the
    # reset were missing entirely.
    original_pump = loop._apump_turn
    observed: list[bool] = []

    async def pump(state: RunState, res: Any, session: Any) -> Any:
        observed.append(state.final_text_is_model_output)
        return await original_pump(state, res, session)

    loop._apump_turn = pump  # type: ignore[method-assign]
    loop.submit("second")
    loop.close()

    assert observed == [False]


# --- the settle coordinator: kernel text installed after a model settle ---------------------


def _settle_coordinator() -> tuple[Any, Any, Any]:
    """The real ``LoopSettleCoordinator`` over stubs, so the assertions run the shipped branch.

    Transcribing the branch into the test instead would pass even with the fix reverted, which is
    the failure mode this whole file exists to avoid.
    """
    from types import SimpleNamespace

    from monoid_agent_kernel.loop_phases import LoopSettleCoordinator

    emitted: list[tuple[str, Any]] = []
    recorder = SimpleNamespace(emit=lambda event_type, **kw: emitted.append((event_type, kw)))
    res = SimpleNamespace(recorder=recorder)
    loop = SimpleNamespace(
        _clear_finish_metadata=lambda context: None,
        _log_finish_observations=lambda state: None,
    )
    return LoopSettleCoordinator(loop), res, emitted


def test_output_contract_fallback_is_not_flagged_as_model_output() -> None:
    """`loop_phases.py` installs kernel text of its own, after the finish path raised the flag.

    A `run.finish` whose summary no validator accepts settles on "Stopped: the final response did
    not satisfy the output contract." — the kernel's sentence, reached with provenance already
    True. Left True, the emit flip would digest the one line explaining why the run stopped.

    This site lives in a different module from the other nine, which is exactly why it shipped
    unpaired: the structural guard used to read only `loop.py`.
    """
    from monoid_agent_kernel.loop import RunState
    from monoid_agent_kernel.loop_phases import _OUTPUT_CONTRACT_STOPPED, SettleDecision

    coordinator, res, _emitted = _settle_coordinator()
    state = RunState()
    state.final_text_is_model_output = True  # as the run.finish path leaves it

    coordinator.apply(
        SettleDecision(kind="exhausted", status="limited", error_code="output_contract_unsatisfied"),
        state,
        res,
        None,
        from_finish=False,
    )

    assert state.final_text == _OUTPUT_CONTRACT_STOPPED
    assert state.final_text_is_model_output is False


def test_an_accepted_model_answer_keeps_its_provenance_through_the_fallback_branch() -> None:
    # The counterweight: the branch installs the kernel sentence only when there is nothing to
    # keep. A model answer that survives validation must stay flagged, or the fix would have
    # traded a leak for text that never leaves the event.
    from monoid_agent_kernel.loop import RunState
    from monoid_agent_kernel.loop_phases import SettleDecision

    coordinator, res, _emitted = _settle_coordinator()
    state = RunState()
    state.final_text = "a real answer"
    state.final_text_is_model_output = True

    coordinator.apply(
        SettleDecision(kind="exhausted", status="limited", error_code="output_contract_unsatisfied"),
        state,
        res,
        None,
        from_finish=False,
    )

    assert state.final_text == "a real answer"
    assert state.final_text_is_model_output is True


def test_a_rejected_finish_summary_drops_its_provenance_with_the_text() -> None:
    # loop_phases.py's second site: the re-prompt path clears the rejected summary, so there is
    # no settled text again until the next turn produces one.
    from monoid_agent_kernel.loop import RunState
    from monoid_agent_kernel.loop_phases import SettleDecision

    coordinator, res, _emitted = _settle_coordinator()
    state = RunState()
    state.final_text = "a rejected summary"
    state.final_text_is_model_output = True

    coordinator.apply(SettleDecision(kind="reprompt"), state, res, None, from_finish=True)

    assert state.final_text == ""
    assert state.final_text_is_model_output is False


# --- restore: provenance is not checkpointed, so it fails closed ---------------------------


def test_restore_fails_closed_on_non_empty_text(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="resumed")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop.restore(RunCheckpoint(run_id=spec.run_id, final_text="something the model may have said"))

    assert loop._session is not None
    assert loop._session.state.final_text == "something the model may have said"
    # Not "we know it was the model's" — "we cannot know, so assume the disclosing case".
    assert loop._session.state.final_text_is_model_output is True


def test_restore_leaves_an_empty_text_unflagged(tmp_path: Path) -> None:
    # The other side of the same rule: fail-closed must not mean "always True", or every resumed
    # run digests an empty string and the flag stops carrying information.
    spec = _spec(tmp_path)
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="resumed")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )
    loop.restore(RunCheckpoint(run_id=spec.run_id, final_text=""))

    assert loop._session is not None
    assert loop._session.state.final_text_is_model_output is False


# --- structural guards: what protects the sites this file cannot enumerate forever ----------


def _unpaired_assignments(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if _ASSIGNMENT.match(line)]
    unpaired: list[str] = []
    for position, index in enumerate(starts):
        # Bound the search by the next assignment so one site cannot be credited with another's
        # flag; a parenthesised multi-line value (the cancel/timeout site) needs the slack.
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        if not any(_PROVENANCE in candidate for candidate in lines[index + 1 : stop]):
            unpaired.append(f"{path.name}:{index + 1}: {lines[index].strip()}")
    return unpaired


def test_every_final_text_assignment_sets_provenance_in_the_same_breath() -> None:
    """A write to ``state.final_text`` that forgets the flag is a silent leak.

    No behavioural test can cover a site that does not exist yet, so the pairing is enforced on
    the source itself: each assignment must set provenance before the next assignment begins.

    Scanned across **all** of ``src/``. The first version of this guard looked only at
    ``loop.py``, and two assignments in ``loop_phases.py`` shipped unpaired underneath it — a
    guard is only as wide as the files it reads.
    """
    modules = sorted(_SRC_ROOT.rglob("*.py"))
    unpaired = [entry for path in modules for entry in _unpaired_assignments(path)]
    assert not unpaired, "final_text assigned without declaring provenance:\n" + "\n".join(unpaired)


def test_the_pairing_guard_actually_reaches_more_than_one_module() -> None:
    """The guard above passes vacuously if its file sweep finds nothing.

    Pinned separately because the failure mode is silent: a wrong root, a changed layout, or a
    stricter pattern all turn the guard into a test that asserts an empty list is empty.
    """
    with_assignments = {
        path.name
        for path in _SRC_ROOT.rglob("*.py")
        if any(_ASSIGNMENT.match(line) for line in path.read_text(encoding="utf-8").splitlines())
    }
    assert {"loop.py", "loop_phases.py"} <= with_assignments, with_assignments


def test_only_the_two_model_authored_sites_set_the_flag_true() -> None:
    """Pins the classification itself, which is the part a reviewer must be able to check.

    ``turn.final_text`` and the ``run.finish`` summary are model prose; everything else the run
    settles on is a kernel string. Flipping any site trips this, including sites whose behavioural
    path is expensive to reach.
    """
    true_sites: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if re.match(r"\s*state\.final_text_is_model_output = True\b", line):
                # Attribute the flag to the assignment it follows.
                preceding = [
                    candidate
                    for candidate in lines[max(0, index - 8) : index]
                    if _ASSIGNMENT.match(candidate)
                ]
                assert preceding, f"{path.name}:{index + 1} sets provenance with no assignment above"
                true_sites.append(preceding[-1].strip())

    assert sorted(true_sites) == sorted(
        [
            "state.final_text = turn.final_text or \"\"",
            "state.final_text = context.pending_finish.summary",
        ]
    )
