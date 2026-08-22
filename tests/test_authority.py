from __future__ import annotations

import threading

import pytest

from monoid_agent_kernel.core.authority import (
    ActivationWriteAuthority,
    WriteAuthorityRevoked,
)
from monoid_agent_kernel.core.cancellation import CancellationToken
from monoid_agent_kernel.core.interruption import InterruptionCause


def test_authority_revoke_is_sticky_idempotent_and_notifies_once() -> None:
    authority = ActivationWriteAuthority()
    calls: list[str] = []
    authority.add_revoke_callback(lambda: calls.append("revoked"))

    assert authority.revoke() is True
    assert authority.revoke() is False
    assert authority.revoked is True
    assert calls == ["revoked"]
    with pytest.raises(WriteAuthorityRevoked):
        authority.assert_active()


def test_late_revoke_callback_runs_immediately() -> None:
    authority = ActivationWriteAuthority()
    authority.revoke()
    calls: list[str] = []

    unsubscribe = authority.add_revoke_callback(lambda: calls.append("late"))
    unsubscribe()

    assert calls == ["late"]


def test_concurrent_revoke_has_one_winner_and_callback_failure_does_not_roll_back() -> None:
    authority = ActivationWriteAuthority()
    callback_calls: list[str] = []

    def broken_callback() -> None:
        callback_calls.append("called")
        raise RuntimeError("diagnostic callback failed")

    authority.add_revoke_callback(broken_callback)
    winners: list[bool] = []
    threads = [threading.Thread(target=lambda: winners.append(authority.revoke())) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)

    assert winners.count(True) == 1
    assert winners.count(False) == 7
    assert callback_calls == ["called"]
    assert authority.revoked is True


def test_local_mutation_linearizes_before_revoke_returns() -> None:
    authority = ActivationWriteAuthority()
    entered = threading.Event()
    release = threading.Event()
    mutated: list[str] = []
    caught: list[BaseException] = []

    def operation() -> None:
        entered.set()
        assert release.wait(5)
        mutated.append("done")

    def guarded_operation() -> None:
        try:
            authority.guard_local_mutation(operation)
        except BaseException as exc:
            caught.append(exc)

    worker = threading.Thread(target=guarded_operation)
    worker.start()
    assert entered.wait(5)
    revoke_finished = threading.Event()
    revoker = threading.Thread(target=lambda: (authority.revoke(), revoke_finished.set()))
    revoker.start()
    assert not revoke_finished.wait(0.05)
    release.set()
    worker.join(5)
    revoker.join(5)

    assert mutated == ["done"]
    assert len(caught) == 1
    assert isinstance(caught[0], WriteAuthorityRevoked)
    assert revoke_finished.is_set()
    with pytest.raises(WriteAuthorityRevoked):
        authority.guard_local_mutation(lambda: mutated.append("late"))
    assert mutated == ["done"]


def test_external_guard_gives_concurrent_revocation_precedence() -> None:
    authority = ActivationWriteAuthority()

    def operation() -> None:
        authority.revoke()
        raise RuntimeError("callback failed")

    with pytest.raises(WriteAuthorityRevoked) as exc_info:
        authority.guard_external_call(operation)
    assert isinstance(exc_info.value.__context__, RuntimeError)


def test_public_cancellation_rejects_lease_loss() -> None:
    token = CancellationToken()

    with pytest.raises(ValueError, match="cancellation cause must be operational"):
        token.cancel(InterruptionCause.LEASE_LOST)


@pytest.mark.parametrize(
    "cause",
    (InterruptionCause.USER_CANCEL, InterruptionCause.GRACEFUL_DRAIN),
)
def test_authority_signal_preserves_first_cancellation_cause(
    cause: InterruptionCause,
) -> None:
    token = CancellationToken()
    authority = ActivationWriteAuthority()
    authority.add_revoke_callback(token._cancel_for_authority_loss)
    token.cancel(cause)

    authority.revoke()

    assert token.cause is cause
    assert authority.revoked is True


def test_authority_signal_wakes_execution_with_lease_loss() -> None:
    token = CancellationToken()
    authority = ActivationWriteAuthority()
    authority.add_revoke_callback(token._cancel_for_authority_loss)

    authority.revoke()

    assert token.snapshot() == (True, InterruptionCause.LEASE_LOST)
