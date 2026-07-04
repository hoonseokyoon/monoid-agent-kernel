from __future__ import annotations

import json
from pathlib import Path

import pytest
from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.lifecycle import (
    LEGAL_TRANSITIONS,
    REASON_TO_STATE,
    TERMINAL_STATES,
    AgentSession,
    LoopSession,
    SessionHealth,
    SessionInspection,
    SessionState,
    assert_transition,
    can_transition,
    session_state_from_run_status,
    session_state_value,
    state_from_suspension,
    to_session_state,
)
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore
from monoid_agent_kernel.core.result import Suspension
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call

# The non-terminal Suspension.reason values the loop can return (see core/result.py).
_NON_TERMINAL_REASONS = ("settled", "awaiting_tasks", "limited", "interrupted", "turn_failed")


def _suspension(reason: str, *, status: str = "completed", error_code: str = "") -> Suspension:
    return Suspension(reason=reason, status=status, error_code=error_code)  # type: ignore[arg-type]


# --- mapping: every reason projects to exactly one state ---------------------------------


@pytest.mark.parametrize("reason", _NON_TERMINAL_REASONS)
def test_each_non_terminal_reason_maps_to_one_state(reason: str) -> None:
    state = state_from_suspension(_suspension(reason))
    assert isinstance(state, SessionState)
    assert REASON_TO_STATE[reason] is state


def test_terminal_reason_maps_to_failed() -> None:
    assert state_from_suspension(_suspension("terminal", status="failed")) is SessionState.FAILED


def test_cancel_terminal_maps_to_cancelled() -> None:
    # Cancel arrives as reason="terminal", status="limited", error_code="cancelled".
    cancelled = _suspension("terminal", status="limited", error_code="cancelled")
    assert state_from_suspension(cancelled) is SessionState.CANCELLED


def test_reason_map_covers_every_non_terminal_reason() -> None:
    # ``paused`` is pre-declared in the map for Step 3 (it is not yet a live reason).
    assert set(REASON_TO_STATE) == set(_NON_TERMINAL_REASONS) | {"paused"}


# --- transition table internal consistency ------------------------------------------------


def test_every_state_is_a_table_key() -> None:
    assert set(LEGAL_TRANSITIONS) == set(SessionState)


def test_every_transition_target_is_a_known_state() -> None:
    for src, targets in LEGAL_TRANSITIONS.items():
        for dst in targets:
            assert dst in SessionState, f"{src} -> {dst} targets an unknown state"


def test_terminal_states_have_empty_out_set() -> None:
    for terminal in TERMINAL_STATES:
        assert LEGAL_TRANSITIONS[terminal] == frozenset()


def test_self_edge_is_always_legal() -> None:
    for state in SessionState:
        assert can_transition(state, state)


def test_assert_transition_raises_on_illegal_edge() -> None:
    with pytest.raises(Exception) as exc_info:
        assert_transition(SessionState.COMPLETED, SessionState.RUNNING)
    assert getattr(exc_info.value, "error_code", "") == "illegal_session_transition"


def test_open_and_close_edges_are_legal() -> None:
    assert can_transition(SessionState.CREATED, SessionState.IDLE)
    assert can_transition(SessionState.IDLE, SessionState.RUNNING)
    assert can_transition(SessionState.AWAITING_INPUT, SessionState.RUNNING)
    assert can_transition(SessionState.AWAITING_INPUT, SessionState.COMPLETED)
    # Every live state can finalize (close/cancel) to a terminal.
    for state in SessionState:
        if state in TERMINAL_STATES:
            continue
        assert can_transition(state, SessionState.COMPLETED)
        assert can_transition(state, SessionState.FAILED)


# --- wire round-trip ----------------------------------------------------------------------


def test_session_state_json_roundtrip() -> None:
    for state in SessionState:
        encoded = json.dumps(state.value)
        assert SessionState(json.loads(encoded)) is state


# --- Step 2: LoopSession facade -----------------------------------------------------------


def _loop(tmp_path: Path, adapter: FakeModelAdapter) -> AgentLoop:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return AgentLoop(
        spec=AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs"),
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )


def test_facade_state_walks_created_idle_running_awaiting(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "a\n"}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ]
    )
    session = LoopSession(_loop(tmp_path, adapter))

    assert session.state is SessionState.CREATED
    session.open()
    assert session.state is SessionState.IDLE
    turn = session.submit("go")
    assert turn.final_text == "done"
    assert session.state is SessionState.AWAITING_INPUT
    result = session.close()
    assert result.status == "completed"
    assert session.state is SessionState.COMPLETED


def test_facade_satisfies_agentsession_protocol(tmp_path: Path) -> None:
    session = LoopSession(_loop(tmp_path, FakeModelAdapter(turns=[])))
    assert isinstance(session, AgentSession)


def test_inspect_and_health_before_open(tmp_path: Path) -> None:
    session = LoopSession(_loop(tmp_path, FakeModelAdapter(turns=[])))
    inspection = session.inspect()
    assert isinstance(inspection, SessionInspection)
    assert inspection.state is SessionState.CREATED
    assert inspection.terminal is False
    assert inspection.pending_tasks is False
    assert inspection.awaiting_task_ids == ()

    health = session.health()
    assert isinstance(health, SessionHealth)
    assert health.alive is True
    # CREATED is not an input-accepting state.
    assert health.can_accept_input is False


def test_inspect_and_health_after_settle(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="hi")])
    session = LoopSession(_loop(tmp_path, adapter))
    session.open()
    session.submit("go")

    inspection = session.inspect()
    assert inspection.state is SessionState.AWAITING_INPUT
    assert inspection.terminal is False
    assert inspection.turn_handle == "r1"
    assert inspection.to_json()["state"] == "awaiting_input"

    health = session.health()
    assert health.alive is True
    assert health.can_accept_input is True
    assert health.to_json()["can_accept_input"] is True


# --- Step 3: pause / resume / cancel ------------------------------------------------------


class _PausingAdapter:
    """Drives the loop and trips the pause flag during the first model turn, so the pause
    lands at the start of step 2 (a clean start-of-step boundary)."""

    def __init__(self, loop_box: list[AgentLoop], turns: list[ModelTurn]) -> None:
        self._loop_box = loop_box
        self._turns = list(turns)
        self.calls = 0

    def next_turn(self, request: object) -> ModelTurn:
        turn = self._turns[self.calls]
        self.calls += 1
        if self.calls == 1:
            self._loop_box[0].pause_turn()
        return turn


def _pausing_session(tmp_path: Path) -> tuple[LoopSession, AgentRunSpec]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    loop_box: list[AgentLoop] = []
    adapter = _PausingAdapter(
        loop_box,
        [
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_write", {"path": "A.md", "content": "a\n"}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="done"),
        ],
    )
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    loop_box.append(loop)
    return LoopSession(loop), spec


def test_pause_lands_at_step_boundary_then_resumes(tmp_path: Path) -> None:
    session, spec = _pausing_session(tmp_path)
    session.open()

    paused = session.run_until_suspended("go")
    assert paused.reason == "paused"
    assert session.state is SessionState.PAUSED
    assert session.inspect().state is SessionState.PAUSED
    assert session.health().alive is True
    # The pause park persisted a checkpoint (snapshot serialized pending_observations).
    cp = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert cp is not None

    # Resume continues the SAME turn (the kept tool observation is re-sent) to settle.
    settled = session.resume()
    assert settled.reason == "settled"
    assert session.state is SessionState.AWAITING_INPUT
    # The paused turn's tool call really ran (read before close tears the session down).
    ws = session.loop._session.res.workspace  # type: ignore[union-attr]
    assert ws.read_bytes("A.md")[0] == b"a\n"

    result = session.close()
    assert result.status == "completed"


def test_session_state_changed_event_emitted_on_pause(tmp_path: Path) -> None:
    session, spec = _pausing_session(tmp_path)
    session.open()
    session.run_until_suspended("go")

    events_path = tmp_path / "runs" / spec.run_id / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").splitlines()
    changed = [json.loads(line) for line in lines if json.loads(line)["type"] == "session.state.changed"]
    assert changed, "expected a session.state.changed event"
    assert changed[-1]["data"]["state"] == "paused"


def test_pause_survives_restart_via_checkpoint(tmp_path: Path) -> None:
    session, spec = _pausing_session(tmp_path)
    session.open()
    paused = session.run_until_suspended("go")
    assert paused.reason == "paused"

    # Fresh "process": read the persisted checkpoint + blobs and restore into a new loop
    # whose adapter serves only the remaining (final) turn.
    record = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert record is not None
    base2 = tmp_path / "workspace2"
    base2.mkdir()
    spec2 = AgentRunSpec(workspace_root=base2, run_root=tmp_path / "runs", run_id=spec.run_id)
    loop2 = AgentLoop(
        spec=spec2,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r2", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config("fs.write")),
    )
    loop2.restore(record.checkpoint, blobs=record.blob)
    resumed = LoopSession(loop2, _state=SessionState.PAUSED)

    settled = resumed.resume()
    assert settled.reason == "settled"
    assert resumed.state is SessionState.AWAITING_INPUT
    # The paused turn's workspace change came back through the restored delta.
    ws = loop2._session.res.workspace  # type: ignore[union-attr]
    assert ws.read_bytes("A.md")[0] == b"a\n"
    loop2.close()


def test_cancel_terminalizes_run(tmp_path: Path) -> None:
    adapter = FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="hi")])
    session = LoopSession(_loop(tmp_path, adapter))
    session.open()
    session.submit("go")
    session.cancel(reason="operator stop")
    # A subsequent pump observes the cancel at the boundary and settles terminal -> CANCELLED.
    suspension = session.run_until_suspended("again")
    assert suspension.reason == "terminal"
    assert suspension.error_code == "cancelled"
    assert session.state is SessionState.CANCELLED
    assert session._cancel_reason == "operator stop"


# --- Step 5: status-vocabulary reconciliation ---------------------------------------------


def test_to_session_state_reconciles_legacy_status_strings() -> None:
    assert to_session_state("queued") is SessionState.CREATED
    assert to_session_state("running") is SessionState.RUNNING
    assert to_session_state("awaiting_input") is SessionState.AWAITING_INPUT
    assert to_session_state("waiting_for_background_jobs") is SessionState.AWAITING_TASKS
    assert to_session_state("completed") is SessionState.COMPLETED
    assert to_session_state("failed") is SessionState.FAILED
    assert to_session_state("limited") is SessionState.LIMITED


def test_to_session_state_folds_cancel_to_cancelled() -> None:
    # The backend records a cancel as status="limited" + error_code="cancelled".
    assert to_session_state("limited", error_code="cancelled") is SessionState.CANCELLED
    assert to_session_state("failed", error_code="cancelled") is SessionState.CANCELLED


def test_session_state_helpers_serialize_and_project_lifecycle_values() -> None:
    assert session_state_value(SessionState.AWAITING_TASKS) == "awaiting_tasks"
    assert session_state_from_run_status("awaiting_tasks") is SessionState.AWAITING_TASKS
    assert session_state_from_run_status("limited", terminal=False) is SessionState.LIMITED
    assert session_state_from_run_status("limited", terminal=True) is SessionState.LIMITED
    assert session_state_from_run_status("limited", error_code="cancelled", terminal=True) is SessionState.CANCELLED


def test_to_session_state_unknown_falls_back_to_created() -> None:
    assert to_session_state("nonsense") is SessionState.CREATED


def test_cancelled_is_terminal() -> None:
    assert SessionState.CANCELLED in TERMINAL_STATES
    assert LEGAL_TRANSITIONS[SessionState.CANCELLED] == frozenset()
