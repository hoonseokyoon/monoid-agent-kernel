from __future__ import annotations

import json
from pathlib import Path

import pytest

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.outcome import InterruptionCause
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.errors import NativeAgentError
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter, fake_tool_call


class _ParkingAdapter(FakeModelAdapter):
    """Requests a pause or an interrupt DURING the first model call — after step 1's
    start-of-step check already passed — so the signal lands at the start of step 2 while
    step 1's tool observation is pending: a genuinely mid-turn park, never a settle."""

    loop_ref: AgentLoop | None = None
    signal: str = "pause"

    def next_turn(self, request):  # noqa: ANN001
        turn = super().next_turn(request)
        if self.loop_ref is not None:
            if self.signal == "pause":
                self.loop_ref.pause_turn()
            else:
                self.loop_ref.interrupt_turn()
        return turn


def _midturn_parked_loop(tmp_path: Path, signal: str) -> tuple[AgentLoop, AgentRunSpec]:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "a.md").write_text("hi\n", encoding="utf-8")
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    adapter = _ParkingAdapter(
        turns=[
            ModelTurn(
                response_id="r1",
                tool_calls=(fake_tool_call("fs_list", {"path": "."}, "c1"),),
            ),
            ModelTurn(response_id="r2", final_text="the real answer"),
        ]
    )
    adapter.signal = signal
    loop = AgentLoop(
        spec=spec,
        model_adapter=adapter,
        runtime_config_provider=runtime_provider(runtime_config("fs.list", "run.finish")),
        cancellation_token=CancellationToken(),
    )
    adapter.loop_ref = loop
    loop.open()
    return loop, spec


def test_cancel_callbacks_are_one_shot_and_removable() -> None:
    token = CancellationToken()
    calls: list[str] = []
    token.add_cancel_callback(lambda: calls.append("kept"))
    remove = token.add_cancel_callback(lambda: calls.append("removed"))
    remove()
    remove()

    token.cancel()
    token.cancel()

    assert calls == ["kept"]
    assert token.requested is True


def test_first_interruption_cause_wins_while_lease_authority_is_independent() -> None:
    token = CancellationToken()

    token.cancel(InterruptionCause.GRACEFUL_DRAIN)
    token.cancel(InterruptionCause.USER_CANCEL)
    token.cancel(InterruptionCause.LEASE_LOST)

    assert token.cause is InterruptionCause.GRACEFUL_DRAIN
    assert token.lease_lost is True


@pytest.mark.parametrize(
    "cause",
    (
        InterruptionCause.PROVIDER_FAILURE,
        InterruptionCause.VALIDATION_FAILURE,
        InterruptionCause.UNKNOWN,
    ),
)
def test_cancellation_token_rejects_failure_causes_without_mutating_state(
    cause: InterruptionCause,
) -> None:
    token = CancellationToken()
    callbacks: list[str] = []
    token.add_cancel_callback(lambda: callbacks.append("called"))

    with pytest.raises(ValueError, match="cancellation cause must be operational"):
        token.cancel(cause)

    assert token.snapshot() == (False, None)
    assert token.lease_lost is False
    assert callbacks == []


def test_cancellation_snapshot_pairs_the_request_with_its_winning_cause() -> None:
    token = CancellationToken()
    assert token.snapshot() == (False, None)

    token.cancel(InterruptionCause.HOST_SHUTDOWN)

    assert token.snapshot() == (True, InterruptionCause.HOST_SHUTDOWN)


def test_callback_added_after_cancellation_runs_immediately() -> None:
    token = CancellationToken()
    token.cancel()
    calls: list[str] = []

    remove = token.add_cancel_callback(lambda: calls.append("late"))
    remove()

    assert calls == ["late"]


def _restorable_loop(tmp_path: Path) -> AgentLoop:
    """A fresh loop over a fresh run dir, built the way a recovery driver builds one.

    Deliberately no ``cancellation_token``: that is the ordinary shape (the constructor
    default is ``None``) and it is exactly the shape the asymmetric restore un-cancelled.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir(parents=True, exist_ok=True)
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    return AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="done")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
    )


def test_a_restored_loop_honors_a_durable_cancellation_without_a_token(tmp_path: Path) -> None:
    """``snapshot()`` wrote the flag unconditionally; the restore applied it conditionally.

    A recovery driver rebuilds the loop from the checkpoint, and a freshly constructed
    ``AgentLoop`` has ``cancellation_token=None`` — so the restore's `and self.cancellation_token
    is not None` guard silently un-cancelled every durably-cancelled run it recovered. The flag is
    the request; the token is only the channel a boundary check reads it through, so the restore
    mints one.
    """
    loop = _restorable_loop(tmp_path)
    assert loop.cancellation_token is None

    loop.restore(RunCheckpoint(run_id=loop.spec.run_id, seq=1, cancellation_requested=True))

    assert loop.cancellation_token is not None
    assert loop.cancellation_token.requested is True
    # ...and the next boundary check actually observes it. The pump catches ``RunCancelled``
    # and settles the run terminal rather than letting it escape, so the observable proof is
    # the park, not the exception type.
    suspension = loop.run_until_suspended("go")
    assert (suspension.reason, suspension.error_code) == ("terminal", "cancelled")


def test_a_restored_loop_leaves_an_uncancelled_run_runnable(tmp_path: Path) -> None:
    """The other half: minting a token is not the same as cancelling one."""

    loop = _restorable_loop(tmp_path)

    loop.restore(RunCheckpoint(run_id=loop.spec.run_id, seq=1, cancellation_requested=False))

    assert loop.cancellation_token is None or loop.cancellation_token.requested is False
    suspension = loop.run_until_suspended("go")
    loop.close()
    assert suspension.reason == "settled"


def test_a_restore_into_an_existing_token_still_cancels_it(tmp_path: Path) -> None:
    """The pre-existing-token path the old guard covered must keep working."""

    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token

    loop.restore(RunCheckpoint(run_id=loop.spec.run_id, seq=1, cancellation_requested=True))

    assert loop.cancellation_token is token
    assert token.requested is True


def test_a_restored_loop_reinstalls_the_durable_interruption_cause(tmp_path: Path) -> None:
    loop = _restorable_loop(tmp_path)

    loop.restore(
        RunCheckpoint(
            run_id=loop.spec.run_id,
            seq=1,
            cancellation_requested=True,
            interruption_cause=InterruptionCause.GRACEFUL_DRAIN.value,
        )
    )

    assert loop.cancellation_token is not None
    assert loop.cancellation_token.cause is InterruptionCause.GRACEFUL_DRAIN
    suspension = loop.run_until_suspended("go")
    assert suspension.reason == "interrupted"
    assert suspension.interruption_cause is InterruptionCause.GRACEFUL_DRAIN
    stored = LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id)
    assert stored is not None
    assert stored.checkpoint.interruption_cause == InterruptionCause.GRACEFUL_DRAIN.value


def test_a_quiescent_snapshot_persists_the_pending_token_cause_atomically(
    tmp_path: Path,
) -> None:
    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token
    loop.open()
    assert loop.run_until_suspended("go").reason == "settled"

    token.cancel(InterruptionCause.GRACEFUL_DRAIN)
    checkpoint = loop.snapshot()
    loop.discard_uncommitted()

    assert checkpoint is not None
    assert checkpoint.cancellation_requested is True
    assert checkpoint.interruption_cause == InterruptionCause.GRACEFUL_DRAIN.value

    restored = _restorable_loop(tmp_path)
    restored.restore(checkpoint)
    try:
        suspension = restored.run_until_suspended(None)
    finally:
        restored.discard_uncommitted()
    assert suspension.reason == "interrupted"
    assert suspension.interruption_cause is InterruptionCause.GRACEFUL_DRAIN


def test_lease_loss_returns_only_an_in_memory_park_and_refuses_close_writes(
    tmp_path: Path,
) -> None:
    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token
    loop.open()
    token.cancel(InterruptionCause.GRACEFUL_DRAIN)
    token.cancel(InterruptionCause.LEASE_LOST)

    suspension = loop.run_until_suspended("go")

    assert token.cause is InterruptionCause.GRACEFUL_DRAIN
    assert token.lease_lost is True
    assert suspension.reason == "interrupted"
    assert suspension.error_code == InterruptionCause.LEASE_LOST.value
    assert suspension.interruption_cause is InterruptionCause.LEASE_LOST
    assert LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id) is None
    with pytest.raises(NativeAgentError, match="lease-lost activation") as exc_info:
        loop.close()
    assert exc_info.value.error_code == "lease_lost"
    assert LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id) is None


def test_token_deadline_uses_the_canonical_timeout_terminal(tmp_path: Path) -> None:
    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token
    loop.open()
    token.cancel(InterruptionCause.DEADLINE)

    suspension = loop.run_until_suspended("go")

    assert suspension.reason == "terminal"
    assert suspension.status == "limited"
    assert suspension.error == "run exceeded max duration"
    assert suspension.error_code == "run_timeout"
    assert suspension.final_text == "Stopped after reaching max duration."
    assert suspension.interruption_cause is InterruptionCause.DEADLINE
    stored = LocalFsCheckpointStore(loop.spec.run_root).latest(loop.spec.run_id)
    assert stored is not None
    assert stored.checkpoint.error_code == "run_timeout"
    assert stored.checkpoint.interruption_cause == InterruptionCause.DEADLINE.value
    loop.discard_uncommitted()


def test_lease_loss_after_a_terminal_verdict_blocks_finalization(tmp_path: Path) -> None:
    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token
    loop.open()
    token.cancel(InterruptionCause.USER_CANCEL)
    terminal = loop.run_until_suspended("go")
    token.cancel(InterruptionCause.LEASE_LOST)

    assert terminal.reason == "terminal"
    assert terminal.interruption_cause is InterruptionCause.USER_CANCEL
    with pytest.raises(NativeAgentError, match="lease-lost activation") as exc_info:
        loop.close()
    assert exc_info.value.error_code == "lease_lost"
    events_path = next(loop.spec.run_root.rglob("events.jsonl"))
    event_types = {
        json.loads(line)["type"]
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert "run.finished" not in event_types


def test_close_promotes_a_cancel_acknowledged_at_a_park(tmp_path: Path) -> None:
    """A cancel that lands while the run sits at a quiescent park has no pump to raise in.

    The mid-run half was always right: a stepping turn hits the boundary check, the pump
    catches ``RunCancelled`` and settles ``status="limited"`` / ``error_code="cancelled"``
    with a terminal park. The parked half read the per-submit reset state at close and
    recorded a clean COMPLETED — and the completed-run cleanup then deleted the very
    checkpoints a cancelled run keeps. ``close()`` now promotes the acknowledged cancel
    through the same vocabulary before finalizing, on both loop halves (``aclose``
    delegates here)."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    token = CancellationToken()
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="first")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        cancellation_token=token,
    )
    loop.open()
    suspension = loop.run_until_suspended("go")
    assert suspension.reason == "settled"

    token.cancel()  # operator cancel while parked; acknowledged, nothing is stepping
    result = loop.close()

    assert (result.status, result.error_code) == ("limited", "cancelled")
    # The settled answer SURVIVES the cancel. v0.20 returned it beside the wrong COMPLETED
    # status; fixing the status must not silently replace the answer with the stop notice —
    # the cancel statement lives in error/error_code. (A mid-turn cancel still gets the
    # notice: its per-submit reset left no settled text to preserve.)
    assert result.final_text == "first"
    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True
    assert stored.checkpoint.cancellation_requested is True


def test_close_routes_a_parked_token_deadline_through_timeout_handling(tmp_path: Path) -> None:
    loop = _restorable_loop(tmp_path)
    token = CancellationToken()
    loop.cancellation_token = token
    loop.open()
    assert loop.run_until_suspended("go").reason == "settled"
    token.cancel(InterruptionCause.DEADLINE)

    result = loop.close()

    assert result.status == "limited"
    assert result.error_code == "run_timeout"
    assert result.final_text == "Stopped after reaching max duration."
    assert result.interruption_cause is InterruptionCause.DEADLINE
    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["type"] for event in events].count("turn.settled") == 1


def test_cancel_acknowledged_at_a_midturn_park_keeps_the_stop_notice(tmp_path: Path) -> None:
    """The other half of the answer-preservation rule: a mid-turn park has no settled text
    (the per-submit reset cleared it), so the promotion still explains the stop."""

    loop, _spec = _midturn_parked_loop(tmp_path, "pause")
    assert loop.run_until_suspended("go").reason == "paused"

    loop.cancellation_token.cancel()
    result = loop.close()

    assert (result.status, result.error_code) == ("limited", "cancelled")
    assert result.final_text == "Stopped because the run was cancelled."


def test_close_of_a_paused_midturn_run_promotes_limited_and_keeps_checkpoints(
    tmp_path: Path,
) -> None:
    """close() of a PAUSED mid-turn run finalized a clean COMPLETED with an empty answer and
    DELETED the checkpoints holding the frozen turn — backend-reachable via pause_run + an
    idle timeout. The close boundary now refuses to record a never-settled turn as a
    success: ``status="limited"`` / ``error_code="closed_unsettled"``, checkpoints kept
    (the delete gates on status=="completed"), classification-empty terminal park."""

    loop, spec = _midturn_parked_loop(tmp_path, "pause")
    suspension = loop.run_until_suspended("go")
    assert suspension.reason == "paused"

    result = loop.close()

    assert (result.status, result.error_code) == ("limited", "closed_unsettled")
    assert result.interruption_cause is None
    assert "interruption_cause" not in result.metrics
    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True
    assert stored.checkpoint.interruption_cause == ""
    last = stored.checkpoint.last_suspension
    assert last is not None and not last.get("interruption_cause")
    events = [
        json.loads(line)
        for line in result.run_dir.joinpath("events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    finished = [event for event in events if event["type"] == "run.finished"]
    assert finished[-1]["data"]["interruption_cause"] == ""
    # This is not a provider failure: the minted terminal park carries no classification.
    last = stored.checkpoint.last_suspension
    assert last is not None and last.get("error_code") == "closed_unsettled"
    assert not last.get("retryable") and not last.get("config_recoverable")
    assert not last.get("provider_error_code") and last.get("http_status") is None


def test_close_of_an_interrupted_park_promotes_limited_and_keeps_checkpoints(
    tmp_path: Path,
) -> None:
    """The interrupted twin: the same close-boundary trace holds (empirically — a stopped
    turn also never settled), so the same promotion binds it."""

    loop, spec = _midturn_parked_loop(tmp_path, "interrupt")
    assert loop.run_until_suspended("go").reason == "interrupted"

    result = loop.close()

    assert (result.status, result.error_code) == ("limited", "closed_unsettled")
    assert result.interruption_cause is None
    assert "interruption_cause" not in result.metrics
    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True
    assert stored.checkpoint.interruption_cause == ""
    last = stored.checkpoint.last_suspension
    assert last is not None and not last.get("interruption_cause")


def test_a_late_cancel_does_not_replace_an_already_chosen_close_verdict(
    tmp_path: Path,
) -> None:
    loop, spec = _midturn_parked_loop(tmp_path, "interrupt")
    assert loop.run_until_suspended("go").reason == "interrupted"
    token = loop.cancellation_token
    assert token is not None
    persist_checkpoint = loop._persist_checkpoint

    def cancel_before_terminal_snapshot(session, suspension):  # noqa: ANN001
        token.cancel(InterruptionCause.USER_CANCEL)
        return persist_checkpoint(session, suspension)

    loop._persist_checkpoint = cancel_before_terminal_snapshot  # type: ignore[method-assign]

    result = loop.close()

    assert (result.status, result.error_code) == ("limited", "closed_unsettled")
    assert result.interruption_cause is None
    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    assert stored.checkpoint.cancellation_requested is False
    assert stored.checkpoint.interruption_cause == ""


def test_a_restored_midturn_park_still_closes_unsettled(tmp_path: Path) -> None:
    """The promotion survives a restart: a restored paused park that is closed without a
    resume promotes exactly as the in-process park would (rehydrated off
    ``last_suspension``, like the turn-failed promotion)."""

    loop, spec = _midturn_parked_loop(tmp_path, "pause")
    assert loop.run_until_suspended("go").reason == "paused"
    loop.release_parked()

    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    restored = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[]),
        runtime_config_provider=runtime_provider(runtime_config("fs.list", "run.finish")),
    )
    restored.restore(stored.checkpoint, blobs=stored.blob)

    result = restored.close()

    assert (result.status, result.error_code) == ("limited", "closed_unsettled")
    assert LocalFsCheckpointStore(spec.run_root).latest(spec.run_id) is not None


def test_a_resumed_pause_that_settles_still_closes_completed(tmp_path: Path) -> None:
    """The counterweight: the marker clears at pump entry, so a pause that is RESUMED and
    settles keeps the clean-completion contract (and its checkpoint cleanup)."""

    loop, spec = _midturn_parked_loop(tmp_path, "pause")
    assert loop.run_until_suspended("go").reason == "paused"
    assert loop.run_until_suspended(None).reason == "settled"

    result = loop.close()

    assert (result.status, result.error_code) == ("completed", "")
    assert result.final_text == "the real answer"
    assert LocalFsCheckpointStore(spec.run_root).latest(spec.run_id) is None


def test_close_without_a_pending_cancel_still_completes_and_cleans_up(tmp_path: Path) -> None:
    """The other half: an uncancelled parked close keeps its clean-completion contract —
    status "completed" and the completed-run checkpoint cleanup."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = AgentRunSpec(workspace_root=workspace, run_root=tmp_path / "runs")
    loop = AgentLoop(
        spec=spec,
        model_adapter=FakeModelAdapter(turns=[ModelTurn(response_id="r1", final_text="first")]),
        runtime_config_provider=runtime_provider(runtime_config("run.finish")),
        cancellation_token=CancellationToken(),
    )
    loop.open()
    assert loop.run_until_suspended("go").reason == "settled"

    result = loop.close()

    assert (result.status, result.error_code) == ("completed", "")
    assert LocalFsCheckpointStore(spec.run_root).latest(spec.run_id) is None
