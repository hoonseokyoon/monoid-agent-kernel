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

_LOOP_SOURCE = Path(__file__).resolve().parents[1] / "src" / "monoid_agent_kernel" / "loop.py"
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


def _final_text_assignments(source: str) -> list[tuple[int, str]]:
    lines = source.splitlines()
    return [
        (index, line)
        for index, line in enumerate(lines)
        if re.match(r"\s*state\.final_text = ", line)
    ]


def test_every_final_text_assignment_sets_provenance_in_the_same_breath() -> None:
    """A future write to ``state.final_text`` that forgets the flag is a silent leak.

    No behavioural test can cover a site that does not exist yet, so the pairing is enforced on
    the source itself: each assignment must set provenance before the next assignment begins.
    """
    source = _LOOP_SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    assignments = _final_text_assignments(source)
    assert assignments, "found no state.final_text assignments — the guard is looking in the wrong file"

    starts = [index for index, _line in assignments]
    unpaired: list[str] = []
    for position, (index, line) in enumerate(assignments):
        # Bound the search by the next assignment so one site cannot be credited with another's
        # flag; a parenthesised multi-line value (the cancel/timeout site) needs the slack.
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        window = lines[index + 1 : stop]
        if not any("state.final_text_is_model_output = " in candidate for candidate in window):
            unpaired.append(f"loop.py:{index + 1}: {line.strip()}")

    assert not unpaired, "final_text assigned without declaring provenance:\n" + "\n".join(unpaired)


def test_only_the_two_model_authored_sites_set_the_flag_true() -> None:
    """Pins the classification itself, which is the part a reviewer must be able to check.

    ``turn.final_text`` and the ``run.finish`` summary are model prose; everything else the run
    settles on is a kernel string. Flipping any site trips this, including sites whose behavioural
    path is expensive to reach.
    """
    source = _LOOP_SOURCE.read_text(encoding="utf-8")
    lines = source.splitlines()
    true_sites: list[str] = []
    for index, line in enumerate(lines):
        if re.match(r"\s*state\.final_text_is_model_output = True\b", line):
            # Attribute the flag to the assignment it follows.
            preceding = [
                candidate
                for candidate in lines[max(0, index - 8) : index]
                if re.match(r"\s*state\.final_text = ", candidate)
            ]
            assert preceding, f"loop.py:{index + 1} sets provenance with no assignment above it"
            true_sites.append(preceding[-1].strip())

    assert sorted(true_sites) == sorted(
        [
            "state.final_text = turn.final_text or \"\"",
            "state.final_text = context.pending_finish.summary",
        ]
    )
