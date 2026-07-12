from __future__ import annotations

from monoid_agent_kernel.core.cancellation import CancellationToken


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
