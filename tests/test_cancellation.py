from __future__ import annotations

from pathlib import Path

from support.runtime import runtime_config, runtime_provider

from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.checkpoint import LocalFsCheckpointStore, RunCheckpoint
from monoid_agent_kernel.core.spec import AgentRunSpec
from monoid_agent_kernel.loop import AgentLoop
from monoid_agent_kernel.providers.base import ModelTurn
from monoid_agent_kernel.providers.fake import FakeModelAdapter


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
    stored = LocalFsCheckpointStore(spec.run_root).latest(spec.run_id)
    assert stored is not None
    assert stored.checkpoint.terminal is True
    assert stored.checkpoint.cancellation_requested is True


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
