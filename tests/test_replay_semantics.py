"""Replay under the loop: determinism, parity, parks, budgets, families, and public surfaces.

W6-4b B4. The adapter and reader are pinned unit-wise beside this file; here the whole engine
drives them, because the properties that matter are loop properties: a recorded run replayed
through ``AgentLoop`` settles the same way, the ledger the replay run writes carries the same
keys line for line, a miss parks a session rather than killing it and promotes to the failure
record only when a one-shot facade closes, budgets trip at the same turn, families replay
across their union, and nothing conversation-shaped reaches a public surface on the way.

The golden fixture (``tests/fixtures/replay_corpus_v1``) is the cross-process half of the
shared-function mutation gate: a corpus recorded under this version's rules, checked in, so a
composition change that forgets to bump the generation turns THIS file red instead of silently
rekeying everyone's corpora.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from support.runtime import runtime_config, runtime_provider, tool_binding

from monoid_agent_kernel.core.agents import AgentRuntimeConfig, PromptSpec, SubagentDefinition
from monoid_agent_kernel.core.model_calls import MODEL_CALLS_FILENAME
from monoid_agent_kernel.core.model_payloads import MODEL_PAYLOADS_FILENAME
from monoid_agent_kernel.core.spec import AgentRunSpec, ModelConfig, RunLimits
from monoid_agent_kernel.errors import TurnNotSettled
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.model_call import ModelCallRunner
from monoid_agent_kernel.providers.base import ModelRequest, ModelTurn, ToolCall
from monoid_agent_kernel.providers.fake import FakeModelAdapter
from monoid_agent_kernel.providers.replay import ReplayModelAdapter

_MARKER = "SECRET-REPLAY-CONTENT-5Z"
_FIXTURE = Path(__file__).parent / "fixtures" / "replay_corpus_v1" / "run-golden"


def _two_turn_adapter(*, reasoning: tuple[dict[str, Any], ...] = ()) -> FakeModelAdapter:
    return FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="resp-1",
                tool_calls=[ToolCall(id="c1", name="fs_list", arguments={"path": "."})],
                usage={"input_tokens": 11, "output_tokens": 5, "total_tokens": 16},
                reasoning=reasoning,
            ),
            ModelTurn(
                response_id="resp-2",
                final_text="done",
                usage={"input_tokens": 21, "output_tokens": 7, "total_tokens": 28},
                stop_reason="stop",
            ),
        ]
    )


def _loop(
    base: Path,
    adapter: Any,
    *,
    limits: RunLimits | None = None,
    record: bool = False,
    **kwargs: Any,
) -> AgentLoop:
    workspace = base / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    spec_kwargs: dict[str, Any] = {"workspace_root": workspace, "run_root": base / "runs"}
    if limits is not None:
        spec_kwargs["limits"] = limits
    return AgentLoop(
        spec=AgentRunSpec(**spec_kwargs),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        model_calls_file=True,
        model_payload_file=record,
        **kwargs,
    )


def _ledger(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / MODEL_CALLS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _events(run_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# --- round-trip determinism and ledger parity --------------------------------------------------


def test_a_recorded_run_replays_to_the_same_answer_and_the_same_keys(tmp_path: Path) -> None:
    """The headline oracle: record a multi-turn run (tool call included), replay it with the
    engine, and the replay run settles on the same final text while its OWN ledger recomputes
    the same request keys line for line -- the runner's recomputation is the proof, no
    instrumentation."""

    instruction = f"List the workspace, then finish. {_MARKER}"
    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once(instruction)
    assert original_result.status == "completed"

    adapter = ReplayModelAdapter([original_result.run_dir])
    replay = _loop(tmp_path / "replay", adapter)
    replay_result = replay.run_once(instruction)

    assert replay_result.status == "completed"
    assert replay_result.final_text == original_result.final_text == "done"
    original_keys = [line["request_digest"] for line in _ledger(original_result.run_dir)]
    replay_keys = [line["request_digest"] for line in _ledger(replay_result.run_dir)]
    assert replay_keys == original_keys
    assert all(line["digest_status"] == "ok" for line in _ledger(replay_result.run_dir))
    assert all(line["error_code"] == "" for line in _ledger(replay_result.run_dir))


def test_a_replay_run_records_into_its_own_directory_never_the_source(tmp_path: Path) -> None:
    """Recording switches compose with replay (the new_episodes shape) -- and the source
    corpus is read-only in fact, not just in intent."""

    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once("go")
    source = original_result.run_dir / MODEL_PAYLOADS_FILENAME
    before = source.read_bytes()

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]), record=True)
    replay_result = replay.run_once("go")

    assert replay_result.status == "completed"
    assert source.read_bytes() == before
    assert (replay_result.run_dir / MODEL_PAYLOADS_FILENAME).exists()


def test_a_token_budget_trips_at_the_same_turn_on_replay(tmp_path: Path) -> None:
    """Replayed usage is summed as real usage (D-e), so a budget that limited the original
    limits the replay at the same point."""

    limits = RunLimits(max_total_tokens=10)
    original = _loop(tmp_path / "original", _two_turn_adapter(), limits=limits, record=True)
    original_result = original.run_once("go")
    assert original_result.status == "limited"

    replay = _loop(
        tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]), limits=limits
    )
    replay_result = replay.run_once("go")

    assert replay_result.status == "limited"
    assert replay_result.error_code == original_result.error_code
    assert [line["usage"] for line in _ledger(replay_result.run_dir)] == [
        line["usage"] for line in _ledger(original_result.run_dir)
    ]


# --- the miss, on every surface a driver actually reads -----------------------------------------


def test_a_one_shot_miss_promotes_to_the_failure_record(tmp_path: Path) -> None:
    """[P10] run_once absorbs the park and close() promotes it: failure.json carries
    replay_miss, checkpoints survive, the exit is a failed result -- the shape every CLI
    user of --replay-from actually sees."""

    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once("recorded conversation")

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]))
    replay_result = replay.run_once(f"a different conversation {_MARKER}")

    assert replay_result.status == "failed"
    assert replay_result.error_code == "replay_miss"
    failure = json.loads((replay_result.run_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["error_code"] == "replay_miss"
    assert (replay_result.run_dir / "checkpoints").exists()


def test_a_session_miss_parks_and_the_session_survives(tmp_path: Path) -> None:
    """Park-not-kill: the session facade raises TurnNotSettled carrying the full
    classification (config_recoverable -- the operator fixes something and resends), the
    session is not terminal, and only close() turns the unrecovered park into the failure
    record."""

    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once("recorded conversation")

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]))
    replay.open()
    with pytest.raises(TurnNotSettled) as caught:
        replay.submit("never recorded")

    suspension = caught.value.suspension
    assert suspension.reason == "turn_failed"
    assert suspension.error_code == "replay_miss"
    assert suspension.config_recoverable is True
    assert suspension.retryable is False

    closed = replay.close()
    assert closed.status == "failed"
    assert closed.error_code == "replay_miss"


def test_a_miss_reaches_public_surfaces_content_free(tmp_path: Path) -> None:
    """The adversarial pin the plan promoted to a contract: the planted conversation marker
    appears in NO public copy of the miss -- not the turn.failed event payload, not
    failure.json, not the result error -- while the classification does."""

    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once("recorded conversation")

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]))
    replay_result = replay.run_once(f"planted {_MARKER} in the instruction")

    failed = [event for event in _events(replay_result.run_dir) if event["type"] == "turn.failed"]
    assert failed, "the recoverable miss must announce itself on the public stream"
    for event in failed:
        payload = json.dumps(event)
        assert _MARKER not in payload
        assert event["data"]["error_code"] == "replay_miss"
        assert event["data"]["config_recoverable"] is True
    assert _MARKER not in (replay_result.error or "")
    assert _MARKER not in (replay_result.run_dir / "failure.json").read_text(encoding="utf-8")


def test_a_missed_call_lands_on_the_replay_runs_ledger_with_its_taxonomy(tmp_path: Path) -> None:
    original = _loop(tmp_path / "original", _two_turn_adapter(), record=True)
    original_result = original.run_once("recorded conversation")

    replay = _loop(tmp_path / "replay", ReplayModelAdapter([original_result.run_dir]))
    replay_result = replay.run_once("never recorded")

    [line] = _ledger(replay_result.run_dir)
    assert line["error_code"] == "replay_miss"
    assert line["provider_error_code"] in {"absent", "identity_mismatch"}
    assert line["config_recoverable"] is True
    assert line["retryable"] is False


# --- D-h shape (b): the refutation pair ----------------------------------------------------------


def test_an_undeclared_reasoning_corpus_replays_only_without_a_declaration(tmp_path: Path) -> None:
    """The counterfactual pair that makes D-h falsifiable end to end. The original adapter
    declared nothing and answered with reasoning, so the loop never re-injected it and the
    recorded second-turn preimage has no reasoning block. The derived (non-declaring) replay
    completes; forcing the declaration the corpus term suggests makes the loop inject a block
    the recorded preimages never had, and turn two misses."""

    reasoning = ({"type": "reasoning", "encrypted_content": "OPAQUE"},)
    original = _loop(tmp_path / "original", _two_turn_adapter(reasoning=reasoning), record=True)
    original_result = original.run_once("go")
    assert original_result.status == "completed"

    derived = _loop(tmp_path / "derived", ReplayModelAdapter([original_result.run_dir]))
    derived_result = derived.run_once("go")
    assert derived_result.status == "completed"
    assert derived_result.final_text == "done"

    forced_adapter = ReplayModelAdapter([original_result.run_dir], provider_name="gateway")
    forced = _loop(tmp_path / "forced", forced_adapter)
    forced_result = forced.run_once("go")
    assert forced_result.status == "failed"
    assert forced_result.error_code == "replay_miss"


# --- the family union ----------------------------------------------------------------------------


_CHILD_MARKER = "CHILD-PERSONA-REPLAY"


class _RoutingAdapter:
    """Parent spawns a child; the child answers by its persona marker (the corpus test's
    routing shape, reused so the recorded family is real)."""

    def __init__(self) -> None:
        self.parent_calls = 0

    def next_turn(self, request: ModelRequest) -> ModelTurn:
        if _CHILD_MARKER in (request.system_prompt or ""):
            return ModelTurn(response_id="child-1", final_text="child done")
        self.parent_calls += 1
        if self.parent_calls == 1:
            return ModelTurn(
                response_id="parent-1",
                tool_calls=(
                    ToolCall(
                        id="spawn-1",
                        name="agent_spawn",
                        arguments={"subagent_type": "child", "prompt": "work"},
                    ),
                ),
            )
        return ModelTurn(response_id="parent-2", final_text="parent done")


def _family_loop(base: Path, adapter: Any, *, record: bool) -> AgentLoop:
    workspace = base / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=base / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(
            AgentRuntimeConfig(
                definition_id="parent",
                prompt=PromptSpec(persona_segments=("PARENT",)),
                tools=(tool_binding("agent.spawn"),),
            )
        ),
        subagent_definitions={
            "child": SubagentDefinition(prompt=PromptSpec(persona_segments=(_CHILD_MARKER,)))
        },
        model_calls_file=True,
        model_payload_file=record,
    )


def _child_ledger_lines(run_root: Path, parent_run_dir: Path) -> list[dict[str, Any]]:
    child_dirs = [
        path.parent
        for path in run_root.glob(f"*/{MODEL_CALLS_FILENAME}")
        if path.parent != parent_run_dir
    ]
    assert len(child_dirs) == 1, "the routing shape spawns exactly one child"
    return _ledger(child_dirs[0])


def test_the_union_replays_the_childs_call_and_the_spawn_observation_is_the_limit(
    tmp_path: Path,
) -> None:
    """What the union actually guarantees, measured rather than hoped (C10 corrected).

    The child records to its own run directory, so the union is a requirement: with it, the
    child's call HITS -- the child shares the parent's adapter instance and its recorded key
    is reachable (witness: the replayed child's own ledger recomputes the original child's
    digest with no error). Without it, the child's call is the miss.

    What the union does NOT buy is the parent's post-spawn turn: the spawn observation the
    model actually saw embeds per-run mints (child_run_id, task_id, traceparent), so the
    replayed parent's next preimage honestly differs -- fabricating the recorded ids to force
    a hit is exactly the invented identity the key doctrine forbids. The family therefore
    fails AT THAT TURN with a diagnosis naming the observation terms, the same shape any
    nondeterministic tool produces; that is a documented v1 limit, and this test is the
    measurement that made the original "union replays the family" oracle false.
    """

    original = _family_loop(tmp_path / "original", _RoutingAdapter(), record=True)
    original_result = original.run_once("delegate")
    assert original_result.status == "completed"
    run_root = original_result.run_dir.parent
    child_dirs = [
        path.parent
        for path in run_root.glob(f"*/{MODEL_PAYLOADS_FILENAME}")
        if path.parent != original_result.run_dir
    ]
    assert len(child_dirs) == 1
    [original_child_line] = _child_ledger_lines(run_root, original_result.run_dir)

    union_adapter = ReplayModelAdapter([original_result.run_dir, child_dirs[0]])
    union = _family_loop(tmp_path / "union", union_adapter, record=False)
    union_result = union.run_once("delegate")

    [union_child_line] = _child_ledger_lines(union_result.run_dir.parent, union_result.run_dir)
    assert union_child_line["request_digest"] == original_child_line["request_digest"]
    assert union_child_line["error_code"] == "", "the child's call replays from the union"
    assert union_result.status == "failed", "the post-spawn parent turn is the v1 limit"
    assert union_result.error_code == "replay_miss"
    assert "observations" in (union_result.error or "")

    parent_only = _family_loop(
        tmp_path / "parent-only",
        ReplayModelAdapter([original_result.run_dir]),
        record=False,
    )
    parent_only_result = parent_only.run_once("delegate")
    [alone_child_line] = _child_ledger_lines(
        parent_only_result.run_dir.parent, parent_only_result.run_dir
    )
    assert alone_child_line["error_code"] == "replay_miss", (
        "without the child's directory, the miss moves INTO the child -- the union is a "
        "requirement, not a convenience"
    )
    assert parent_only_result.status == "failed"


# --- the golden fixture --------------------------------------------------------------------------


def _golden_request() -> ModelRequest:
    return ModelRequest(
        instruction="golden instruction",
        system_prompt="golden system",
        tools=(),
        model=ModelConfig(model="golden-model", provider="golden-provider"),
    )


def test_the_golden_corpus_still_replays_under_todays_rules() -> None:
    """The cross-version half of the shared-function gate. This corpus was recorded by the
    shipped writer under the current generation; replaying it proves today's composition
    still derives the recorded key. A composition change turns this red -- which is the
    demand that it either restore compatibility or bump the generation and disown the
    corpus deliberately (regenerate the fixture in the same commit, and say so)."""

    adapter = ReplayModelAdapter([_FIXTURE])
    turn, receipt = asyncio.run(ModelCallRunner(adapter=adapter).acall(_golden_request()))

    assert turn.final_text == "the golden answer"
    assert turn.response_id == "golden-1"
    assert (
        receipt.request_digest == "a0c033b275e133b758b37aa618b1671e2f0dacc0b81fd578fc3526d8e3a718a7"
    )
