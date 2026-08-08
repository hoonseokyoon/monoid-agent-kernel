"""Is the replay's own recording the same corpus as the one it replayed from?

W6-4b. The suites beside this one pin the reader, the adapter and the loop a route at a time;
this one asks the question that does not name a route. Record live into A, replay A while
recording into B, and compare the corpora. A cursor that slips changes which answer a call
receives, which changes the next request, which changes B -- so the check reddens on the whole
failure class rather than on the six ways into it that have been found so far.

Two tiers, because they answer different questions. **Tier A** asserts B *is* A modulo the run's
own identity: the regression net a refactor of the settle seam has to pass. **Tier B** doctors A
so a slot is unusable and asserts only that no recorded answer was served for a call it does not
belong to -- the substitution itself, tolerant of a fallthrough having served the call live.

Tier B runs at the ``ModelCallRunner`` level rather than through ``AgentLoop``, and that is a
measured decision, not a convenience. Through the loop the re-attempt of a parked turn does not
recompute the same key -- ``instruction`` is only in the preimage when ``pending_user_input`` is
set, and the failing turn consumed it (the third irreproducible shape in ``docs/CLI.md``) -- so
the re-attempt asks a *different* key and never reaches the cursor that slipped. Verified by
reverting the two-phase consume and observing a byte-identical gen-2: at the loop level these
fixtures would pass against a known-broken tree, which is worse than not having them.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from support.replay_oracle import (
    MASKED,
    alignment_report,
    assert_no_substitution,
    assert_pure_replay_equivalent,
    assert_supply_conserved,
    read_corpus,
    structural_diff,
)
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.model_payloads import (
    MODEL_PAYLOADS_FILENAME,
    PAYLOAD_OFFLOAD_THRESHOLD_BYTES,
    model_response_record,
)
from monoid_agent_kernel.core import payload_replay
from monoid_agent_kernel.core.payload_replay import ReplayCorpus
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import ModelAdapterError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import ModelCallRunner, SettledModelCall
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, ToolCall
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.providers.replay import ReplayModelAdapter
from monoid_agent_kernel.recorder import AgentRecorder

_MARKER = "SECRET-EQUIVALENCE-7Q"


# --- harness -----------------------------------------------------------------------------------


def _loop(base: Path, adapter: Any, *, record: bool = False, **kwargs: Any) -> AgentLoop:
    workspace = base / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=base / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_calls_file=True,
        model_payload_file=record,
        **kwargs,
    )


class _Scripted:
    """The recording-side adapter, and the replaying-side inner when a test needs one."""

    def __init__(self, turns: list[Any]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        del request
        self.calls += 1
        turn = self.turns.pop(0)
        if isinstance(turn, BaseException):
            raise turn
        return turn


def _record_calls(
    base: Path, adapter: Any, requests: list[ModelRequest], *, run_id: str = "run-1"
) -> tuple[Path, list[str]]:
    """Drive ``adapter`` through the real runner and recorder; return its run dir and keys."""

    recorder = AgentRecorder(
        base / "runs",
        run_id,
        status_file=False,
        model_calls_file=True,
        model_payload_file=True,
    )
    digests: list[str] = []

    def sink(call: SettledModelCall) -> None:
        digests.append(call.receipt.request_digest)
        recorder.record_settled_call(call)

    runner = ModelCallRunner(adapter=adapter, settled_sink=sink, capture_request_preimage=True)
    for request in requests:
        try:
            asyncio.run(runner.acall(request))
        except (ModelAdapterError, RuntimeError, ValueError):
            pass
    recorder.close()
    return base / "runs" / run_id, digests


def _request(text: str = "hi") -> ModelRequest:
    return ModelRequest(instruction=text, system_prompt="sys", tools=())


def _make_unusable(
    run_dir: Path, digest: str, *, body: Any = None, reason: str = "too_large"
) -> None:
    """Put an unusable answer in front of the recorded ones, under the same key.

    Slot 0 becomes a refusal the cursor must not advance past; the real answers move to slots 1
    and up, where serving one at position 0 is exactly the substitution the oracle names.
    """

    path = run_dir / MODEL_PAYLOADS_FILENAME
    refused = json.dumps(
        model_response_record(
            body,
            call_index=0,
            request_digest=digest,
            unrecorded_reason=reason,
            run_id="doctored",
            root_run_id="doctored",
            recorded_at="2026-08-08T00:00:00Z",
        ),
        sort_keys=True,
    )
    out: list[str] = []
    inserted = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not inserted and json.loads(line).get("kind") == "model_response":
            out.append(refused)
            inserted = True
        out.append(line)
    assert inserted, "the fixture recorded no answer to stand in front of"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _two_turn_adapter() -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="resp-1",
                tool_calls=[ToolCall(id="c1", name="fs_list", arguments={"path": "."})],
                usage={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
            ),
            ModelTurn(
                response_id="resp-2",
                final_text="done",
                usage={"input_tokens": 21, "output_tokens": 7, "total_tokens": 28},
                stop_reason="stop",
            ),
        ]
    )


# --- Tier A: the replay reproduces its source ---------------------------------------------------


def test_a_recorded_run_and_its_replay_are_one_corpus(tmp_path: Path) -> None:
    """The regression net. A multi-turn run with a tool call, replayed with recording on,
    produces a corpus identical to its source but for the run's own identity."""

    instruction = f"List the workspace, then finish. {_MARKER}"
    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    source = original.run_once(instruction)
    assert source.status == "completed"

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([source.run_dir]), record=True)
    replayed = replay.run_once(instruction)
    assert replayed.status == "completed"

    assert_pure_replay_equivalent(source.run_dir, replayed.run_dir)


def test_an_offloaded_answer_lands_offloaded_on_replay_too(tmp_path: Path) -> None:
    """Inline-vs-offloaded placement is a pure function of encoded length, so a faithful replay
    cannot move a body across the threshold. Asserted through the same masked equality, on a
    body big enough that the question is live."""

    big = "x" * (PAYLOAD_OFFLOAD_THRESHOLD_BYTES + 1024)
    original = _loop(
        tmp_path / "original",
        FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text=big, stop_reason="stop")]),
        record=True,
    )
    source = original.run_once("say the long thing")
    assert source.status == "completed"

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([source.run_dir]), record=True)
    replayed = replay.run_once("say the long thing")
    assert replayed.status == "completed"

    view = read_corpus(source.run_dir)
    assert any(a.placement == "reference" for q in view.answers.values() for a in q), (
        "the fixture is degenerate: nothing was offloaded, so placement is not under test"
    )
    assert_pure_replay_equivalent(source.run_dir, replayed.run_dir)


# --- Tier B: no answer is served for a call it does not belong to -------------------------------


def test_a_standing_refusal_is_re_earned_never_replaced(tmp_path: Path) -> None:
    """Route 1. A refusal must leave the cursor where it was, so the idempotent re-attempt
    earns the same refusal rather than the answer belonging to the call after it."""

    source, digests = _record_calls(
        tmp_path / "original",
        _Scripted(
            [
                ModelTurn(response_id="r-1", final_text="A-one"),
                ModelTurn(response_id="r-2", final_text="A-two"),
            ]
        ),
        [_request(), _request()],
    )
    assert digests[0] == digests[1], "two identical calls must be one key with two answers"
    _make_unusable(source, digests[0])

    replay_dir, _ = _record_calls(
        tmp_path / "replay",
        ReplayModelAdapter([source]),
        [_request(), _request()],
        run_id="run-2",
    )

    assert_no_substitution(source, replay_dir)


def test_a_live_serve_that_failed_does_not_move_the_sequence(tmp_path: Path) -> None:
    """Routes 2 and 3. The slot is spent only when the inner actually served; an inner that
    raises parks the turn, and the re-attempt must not be handed the next recording."""

    source, digests = _record_calls(
        tmp_path / "original",
        _Scripted(
            [
                ModelTurn(response_id="r-1", final_text="A-one"),
                ModelTurn(response_id="r-2", final_text="A-two"),
            ]
        ),
        [_request(), _request()],
    )
    assert digests[0] == digests[1]
    _make_unusable(source, digests[0])

    failing = _Scripted([RuntimeError("the live call failed"), RuntimeError("and again")])
    replay_dir, _ = _record_calls(
        tmp_path / "replay",
        ReplayModelAdapter([source], inner=failing),
        [_request(), _request()],
        run_id="run-2",
    )

    assert failing.calls == 2, "both calls must have reached the inner"
    assert_no_substitution(source, replay_dir)


def test_a_held_slot_comes_back_when_the_inner_raised(tmp_path: Path) -> None:
    """Route 3, and it takes the *fallthrough* exit to reach.

    A record ``consume`` handed over and reconstruction then rejected is held. With no inner
    the wrapper settles it on the way to raising ``ReplayMiss``; the half that broke is the
    other one, where a live inner was asked and raised, so the fixture has to have an inner.
    """

    source, digests = _record_calls(
        tmp_path / "original",
        _Scripted(
            [
                ModelTurn(response_id="r-1", final_text="A-one"),
                ModelTurn(response_id="r-2", final_text="A-two"),
            ]
        ),
        [_request(), _request()],
    )
    _make_unusable(
        source,
        digests[0],
        reason="",
        body={
            "response_id": "r-bad",
            "final_text": None,
            "tool_calls": [{"id": "c1", "name": "fs_list"}],  # no arguments: not a triple
            "reasoning": [],
            "usage": {},
            "stop_reason": None,
            "provider_retried": False,
        },
    )

    failing = _Scripted([RuntimeError("the live call failed"), RuntimeError("and again")])
    replay_dir, _ = _record_calls(
        tmp_path / "replay",
        ReplayModelAdapter([source], inner=failing),
        [_request(), _request()],
        run_id="run-2",
    )

    assert failing.calls == 2, "both calls must have reached the inner"
    assert_no_substitution(source, replay_dir)


def test_a_sync_inner_returning_an_awaitable_does_not_move_the_sequence(tmp_path: Path) -> None:
    """Route 6, at corpus level. An inner that hands back an awaitable has done no provider
    work, so nothing may be settled on its behalf -- and no declaration-side gate can see the
    shape, which is why the wrapper asks the result."""

    source, digests = _record_calls(
        tmp_path / "original",
        _Scripted(
            [
                ModelTurn(response_id="r-1", final_text="A-one"),
                ModelTurn(response_id="r-2", final_text="A-two"),
            ]
        ),
        [_request(), _request()],
    )
    _make_unusable(source, digests[0])

    class _AwaitableInner:
        def next_turn(self, request: ModelRequest) -> Any:
            del request

            async def _later() -> ModelTurn:
                raise RuntimeError("the live call failed after the wrapper returned")

            return _later()

    replay_dir, _ = _record_calls(
        tmp_path / "replay",
        ReplayModelAdapter([source], inner=_AwaitableInner()),
        [_request(), _request()],
        run_id="run-2",
    )

    assert_no_substitution(source, replay_dir)


# --- route 4 has its own oracle, because this one cannot see it ---------------------------------


def test_one_directory_named_twice_supplies_its_answers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route 4 is a supply-multiplicity defect, invisible to the equivalence oracle: duplicate
    answers pile up behind the cursor and are never asked for, so the replay's corpus comes out
    identical either way. Counted from the files instead, independently of the reader.

    The inode fallback is forced, because that is the arm the defect lives on -- on a volume
    that proves an inode the identity never reaches the path composition at all, and a fixture
    that does not force it would pass against any fallback whatsoever.
    """

    source, _ = _record_calls(
        tmp_path / "original",
        _Scripted([ModelTurn(response_id="r-1", final_text="A-one")]),
        [_request()],
    )
    monkeypatch.setattr(
        payload_replay,
        "file_identity",
        lambda metadata: payload_replay.VerifiedFileIdentity(device=0, inode=0),
    )

    # Textually DIFFERENT spellings of one directory, because a fixture whose "two names" are
    # the same string only drives the repeat any fallback collapses -- the fixture defect the
    # mutation axis found in this repair's first pin.
    #
    # All three are ``..`` round trips, which every filesystem resolves. A case-flipped spelling
    # is added only where it actually names the directory: unprobed, it made this test pass here
    # and error on every Linux CI job, because `ReplayCorpus.load` refuses an absent source at
    # construction and `platform-smoke` -- the Windows/macOS job -- runs a fixed file list that
    # excludes this suite. A fixture that only holds on the author's filesystem is the same
    # defect as a rule that only holds on one of two branches.
    parent, grandparent = source.parent, source.parent.parent
    spellings = [
        source,
        parent / ".." / parent.name / source.name,
        parent / ".." / ".." / grandparent.name / parent.name / source.name,
    ]
    flipped = parent / source.name.upper()
    if flipped != source and flipped.exists():
        spellings.append(flipped)
    assert len({str(path) for path in spellings}) == len(spellings) >= 3, (
        "the spellings must differ as text"
    )
    corpus = ReplayCorpus.load(spellings)

    assert_supply_conserved(spellings, corpus)


# --- the oracle's own gate: an oracle with no negative pin is unfalsifiable ----------------------


def test_the_oracle_reports_a_corpus_that_is_not_a_replay_of_its_source(tmp_path: Path) -> None:
    """Kills the ``return []`` mutant: a run of a *different* conversation must not compare
    equal to the source it did not replay."""

    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    source = original.run_once(f"the recorded conversation {_MARKER}")
    other = _loop(tmp_path / "other", _two_turn_adapter(), record=True)
    unrelated = other.run_once(f"a completely different conversation {_MARKER}")

    assert structural_diff(read_corpus(source.run_dir), read_corpus(unrelated.run_dir)), (
        "the comparator called two different conversations one corpus"
    )
    with pytest.raises(AssertionError):
        assert_pure_replay_equivalent(source.run_dir, unrelated.run_dir)


def test_the_oracle_reports_a_shifted_answer(tmp_path: Path) -> None:
    """Kills the 'never emit a Slip' mutant, by building the substitution by hand rather than
    hoping a fixture produces one."""

    source, digests = _record_calls(
        tmp_path / "original",
        _Scripted(
            [
                ModelTurn(response_id="r-1", final_text="A-one"),
                ModelTurn(response_id="r-2", final_text="A-two"),
            ]
        ),
        [_request(), _request()],
    )
    assert digests[0] == digests[1]

    shifted = tmp_path / "shifted"
    shifted.mkdir()
    lines = (source / MODEL_PAYLOADS_FILENAME).read_text(encoding="utf-8").splitlines()
    responses = [
        i for i, line in enumerate(lines) if json.loads(line).get("kind") == "model_response"
    ]
    assert len(responses) == 2
    first, second = responses
    lines[first], lines[second] = lines[second], lines[first]
    (shifted / MODEL_PAYLOADS_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    slips = alignment_report(read_corpus(source), read_corpus(shifted))
    assert [(s.served_position, s.source_slot) for s in slips] == [(0, 1), (1, 0)], (
        f"a swapped pair must be reported as two slips, got {slips}"
    )
    with pytest.raises(AssertionError):
        assert_no_substitution(source, shifted)


def test_the_mask_is_exactly_the_three_fields_a_replay_may_differ_in() -> None:
    """Pinned to the literal names the module documents, not to the tuple the code holds --
    a test that reads its expectation out of the symbol under test passes whatever the code
    does, which is how a 12-character digest bound stayed green at 64."""

    assert MASKED == ("run_id", "root_run_id", "recorded_at")
